//! `policy --sim` — 学習済み方策を MuJoCo（misa-plant-mujoco）で閉ループに
//! 回す。実機と**同じ** NaturalController・同じ脚オドメトリ・同じデコードを
//! 通すので、ここが歩けば「実機に送るバイトと同じ計算」が歩いている。
//! 違うのは Plant だけ（DDS ↔ MuJoCo）。
//!
//! Python の `sim2sim_mit_go2_mujoco.py --natural-walk` と同じ構造
//! （物理 2 ms × 10 = 50 Hz 推論、ZOH、毎物理周期の τ_ff）だが、物理は
//! articara の MJCF 出力（go2.misa の armature 0.01 / damping 2.0 /
//! effort 23.7・45.43 を含む）で、あちらの go2.xml とは別実装。
//!
//! ```text
//! export MUJOCO_DYNAMIC_LINK_DIR=$HOME/.mujoco/mujoco-3.8.0/lib
//! cargo run --release --features sim -- policy --sim \
//!   --model .../exported/policy.onnx --viz --viz-endpoint tcp/127.0.0.1:7447
//! ```
//!
//! articara（`cargo run --release --features viz`、モデルに go2.misa を
//! 開く）の Live gait feed で同じキー / エンドポイントを入れると、指令
//! （planned、ゴースト）と MuJoCo の実測（measured）が重なって描かれる。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use misa_plant_mujoco::{MujocoPlant, SimOptions};
use misa_policy_runner::go2::{dc_motor_clip, effort_limit_isaac, GO2_TO_ISAAC};
use misa_policy_runner::{ObsInput, PolicyTick};
use misa_runner::jointvec::JointVec;
use misa_runner::misa_core::{AxisId, Command, ControlMode, Observation, Plant};
use misa_runner::viz;

use crate::estimator::LegOdometry;
use crate::go2_plant::{go2_axes, MISA_TO_GO2};
use crate::policy::{cmd_clamp, spawn_keyboard, world_to_body, Args, Ctl};

const CONTROL_DT: f64 = 0.002;
const DECIMATION: u64 = 10;
/// 転倒判定: 高さがここを割るか、roll/pitch がここを超えたら止める。
const FALL_HEIGHT_M: f64 = 0.12;
/// この sim（go2.misa、粘性 2.0）で安定に歩ける |vx| の実測上限より一段下。
/// Pure 契約は 0.7 まで歩き 0.8 で転倒したので、余裕を見て 0.6。
const SIM_SAFE_VX_MAX: f64 = 0.6;
/// 受動粘性の既定 [N·m·s/rad]。Unitree 公式 MuJoCo モデルの値。
const DEFAULT_JOINT_DAMPING: f64 = 0.1;
const FALL_TILT_RAD: f64 = 1.0;

/// misa 軸 i（脚順 FL,FR,RL,RR × h/t/c）→ Isaac index（型順）。
fn misa_to_isaac(i: usize) -> usize {
    (i % 3) * 4 + i / 3
}

/// rpy（ZYX オイラー角）→ クォータニオン (w,x,y,z)。
/// MujocoPlant の `Imu` は rpy しか運ばないので、観測の組み立て側で戻す。
fn quat_from_rpy(rpy: [f64; 3]) -> [f64; 4] {
    let q = nalgebra::UnitQuaternion::from_euler_angles(rpy[0], rpy[1], rpy[2]);
    [q.w, q.i, q.j, q.k]
}

/// `tcp/127.0.0.1:7447` のような zenoh エンドポイントから `host:port` を取る。
fn socket_addr_of(endpoint: &str) -> Option<String> {
    let rest = endpoint.split_once('/').map(|(_, r)| r).unwrap_or(endpoint);
    let addr = rest.split(['?', '#']).next().unwrap_or(rest);
    addr.contains(':').then(|| addr.to_string())
}

/// 待ち受けポートが空いているかを先に見る。埋まっていたら、誰が掴んで
/// いるかの調べ方と逃げ道まで書いたエラーを返す（zenoh 由来の
/// "Address already in use" だけだと対処が分からない）。
fn check_viz_port(endpoint: &str) -> Result<(), String> {
    let Some(addr) = socket_addr_of(endpoint) else {
        return Ok(()); // tcp 以外（udp/unixsock など）は判定しない
    };
    match std::net::TcpListener::bind(&addr) {
        Ok(l) => {
            drop(l);
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Err(format!(
            "viz の待ち受け {endpoint} は既に使われています（{addr}）。\n\
             go2-run 側が listen する側なので、このポートは空いている必要があります。\n\
             よくある原因: (1) ROS 2 の rmw_zenoh ルータ rmw_zenohd が 7447 を使っている\n\
             （zenoh ルータの既定ポート。ROS 2 を使うなら止めずに別番号へ逃げる）、\n\
             (2) 前回の go2-run がまだ生きている、(3) articara の Live feed を Connect では\n\
             なく Listen にしている。\n\
             誰が掴んでいるか:  ss -tlnp | grep {port}    （または lsof -i :{port}）\n\
             別ポートで逃げる:  GO2_VIZ_ENDPOINT=tcp/127.0.0.1:{next_port} ./scripts/policy_sim.sh …\n\
             （articara 側の endpoint も同じ番号に合わせる）",
            port = addr.rsplit(':').next().unwrap_or("7447"),
            next_port = addr
                .rsplit(':')
                .next()
                .and_then(|p| p.parse::<u16>().ok())
                .map(|p| p.saturating_add(1))
                .unwrap_or(7448),
        )),
        // bind できない他の理由（権限など）は zenoh 側に任せる
        Err(_) => Ok(()),
    }
}

/// `--joint-damping` 用に .misa の受動粘性を書き換えた一時コピーを作る。
///
/// go2.misa の `damping = 2.0` は MuJoCo Menagerie の go2.xml
/// （`<joint damping="2" armature="0.01" frictionloss="0.2"/>`）由来で、
/// 数値安定性を優先した保守的な値。関節速度 10 rad/s で 20 N·m を食う計算に
/// なり、実機の Go2 が 2.5 m/s 以上出せる事実と両立しない。学習側（Isaac）は
/// 受動粘性ゼロで、V50/V100 は U(0,2) でランダム化して両端に耐えさせている。
/// ここを実機寄り（0.1〜0.5 程度）にすると sim でも高速側が出る。
fn misa_with_damping(misa_path: &str, damping: f64) -> Result<String, String> {
    let text =
        std::fs::read_to_string(misa_path).map_err(|e| format!("{misa_path} を読めません: {e}"))?;
    let mut out = String::with_capacity(text.len());
    let mut hits = 0usize;
    // go2.misa には `*_foot_fixed`（type = "fixed"、damping = 0）も damping の
    // 行を持つ。固定ジョイントに自由度は無いので書き換えても物理は変わらない
    // が、件数の表示が誤解を招くので可動関節だけを対象にする。ブロック内の
    // 並びは name → type → …→ damping なので、直近の type を見れば判る。
    let mut in_fixed = false;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("[[") {
            in_fixed = false;
        } else if t.starts_with("type") && t.contains('=') {
            in_fixed = t.contains("\"fixed\"");
        }
        if !in_fixed && t.starts_with("damping") && line.contains('=') {
            let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
            out.push_str(&format!("{indent}damping = {damping}\n"));
            hits += 1;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if hits == 0 {
        return Err(format!("{misa_path} に damping の行が見つかりません"));
    }
    let stem = std::path::Path::new(misa_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let path = std::env::temp_dir().join(format!("{stem}_damp{damping}.misa"));
    std::fs::write(&path, out).map_err(|e| format!("一時 .misa を書けません: {e}"))?;
    // メッシュは .misa からの相対パスで引かれるので、元と同じディレクトリに
    // 置けない場合は解決できない。そこで元ディレクトリに置き直す。
    let side = std::path::Path::new(misa_path)
        .parent()
        .map(|d| d.join(format!(".{stem}_damp{damping}.misa")));
    if let Some(side) = side {
        if std::fs::copy(&path, &side).is_ok() {
            eprintln!(
                "policy-sim: 受動粘性を {damping} N·m·s/rad に差し替えました（{hits} 関節、\
                 {} を使用）",
                side.display()
            );
            return Ok(side.to_string_lossy().into_owned());
        }
    }
    Ok(path.to_string_lossy().into_owned())
}

pub(crate) fn run(a: &Args) -> Result<(), String> {
    let mut ctl = Ctl::load(a)?;
    if a.odom_calibrated && !ctl.wants_velocity() {
        return Err("--odom-calibrated は76入力Pureポリシー専用です".into());
    }
    // Pure 契約は 0.30 m のしゃがみ姿勢で学習されている（doc/mit_pure.md）。
    let body_height = a.cfg.body_height.unwrap_or(if matches!(ctl, Ctl::Pure(_)) {
        0.30
    } else {
        0.40
    });
    eprintln!(
        "policy-sim: {}（契約: {}、stride {:.2}/{:.2}, height {:.2} m）を {} で回します",
        a.model,
        ctl.name(),
        a.cfg.stride_gain,
        a.cfg.yaw_stride_gain.unwrap_or(a.cfg.stride_gain),
        body_height,
        a.misa,
    );

    // 既定姿勢（ik(trajectory(0,0))）から物理を始める — Python の
    // strict-flat と同じ。base は歩行高さに置き、足球（半径 0.023 m）が
    // ちょうど接地する。
    let axes = go2_axes()?;
    let default_isaac = ctl.default_pose_isaac();
    let home: Vec<(String, f64)> = axes
        .axes()
        .iter()
        .enumerate()
        .map(|(i, ax)| (ax.name.clone(), default_isaac[misa_to_isaac(i)]))
        .collect();
    // go2.misa の damping = 2.0 は MuJoCo Menagerie 由来で、**Unitree 公式の
    // MuJoCo モデル（unitree_mujoco の unitree_robots/go2/go2.xml）は 0.1**。
    // armature 0.01 / frictionloss 0.2 / ctrlrange 23.7・45.43 / cone elliptic /
    // impratio 100 は両者一致で、damping だけが 20 倍違う。学習側（Isaac）も
    // 受動粘性ゼロなので、既定はメーカー値に寄せる。Menagerie の保守値で
    // 頑健性を見たいときは --joint-damping 2.0。
    let damping = a.joint_damping.unwrap_or(DEFAULT_JOINT_DAMPING);
    let misa_path = misa_with_damping(&a.misa, damping)?;
    let opts = SimOptions {
        misa_path: misa_path.clone(),
        control_period_s: CONTROL_DT,
        timestep_s: Some(CONTROL_DT),
        // Impedance 指令が毎周期 kp/kd を運ぶので、ここの既定はランプ前の
        // 保持にしか使われない。名目ゲインにしておく。
        actuator_kp: ctl.initial_gains().0,
        actuator_kv: ctl.initial_gains().1,
        torque_scale: 1.0,
        velocity_kv: 20.0,
        base_height_m: body_height,
        home,
        // 接触モデルは Python の参照プラント（go2-gait-runner の go2.xml、
        // `<option cone="elliptic" impratio="100"/>` と足 geom の
        // friction="0.8 0.02 0.01"）に合わせる。既定の pyramidal /
        // impratio=1 では**接地足が荷重の下で滑る**（misa-plant-mujoco 自身の
        // SimOptions::impratio のコメントどおり）。実測では cmd 0.7 で横に
        // 4.5 m 流れ、脚オドメトリが真値の 1/3 しか出ず（滑りの分だけ足が
        // 空回りする）、1.0 では 1〜3 s で転倒していた。粘性を 0.1 まで
        // 下げても直らなかったので、原因は粘性ではなく接触側。
        friction: Some(a.friction.map_or([0.8, 0.02, 0.01], |mu| [mu, 0.02, 0.01])),
        impratio: Some(a.impratio.unwrap_or(100.0)),
        cone: Some(a.cone.clone().unwrap_or_else(|| "elliptic".into())),
        contact_threshold_n: 5.0,
        feet: ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        root_link: "base".into(),
    };
    let mut plant = MujocoPlant::new(axes, &opts)?;
    let mut obs = Observation::empty(12, 4);
    let mut cmd_out = Command::idle(12);

    // articara へのライブ配信。--viz-endpoint は**待ち受け**なので、ポートが
    // 空いていることを先に確かめる（zenoh の失敗メッセージは長くて原因と
    // 対処が読み取りにくい）。
    if a.viz {
        if let Some(ep) = a.viz_endpoint.as_deref() {
            check_viz_port(ep)?;
        }
    }
    let mut publisher = if a.viz {
        let cfg = viz::VizConfig {
            enabled: true,
            endpoint: a.viz_endpoint.clone(),
            rate_hz: a.viz_rate_hz,
            ..viz::VizConfig::default()
        };
        let p = viz::Publisher::new(&cfg)?;
        eprintln!(
            "policy-sim: viz を {} / {} へ配信します（{} Hz）",
            viz::VizConfig::default().key_planned,
            viz::VizConfig::default().key_measured,
            a.viz_rate_hz
        );
        Some(p)
    } else {
        None
    };

    // 最初の 1 tick は指令 Idle（その場保持）で回して観測を得る。
    plant.exchange(&cmd_out, &mut obs)?;

    // go2.misa は粘性ダンピング 2.0 の**悲観**プラント。ここで出せる速度は
    // 実機よりかなり低く（実測: Pure 契約は vx 0.7 まで歩き、0.8 で転倒）、
    // テレオペで W を押し続けると学習域の上限まで上がって転ぶ。--vx-max
    // 指定が無ければ、この sim では安全側に抑える（実機側は抑えない）。
    // 粘性を実機寄りに下げてあるなら、悲観プラント向けの上限は要らない。
    let lowered_damping = damping < 1.0;
    let vx_max = a.vx_max.or_else(|| {
        let trained = ctl.trained_vx_max();
        (!lowered_damping && trained > SIM_SAFE_VX_MAX).then(|| {
            eprintln!(
                "policy-sim: この sim（go2.misa、粘性 2.0 の悲観プラント）では \
                 |vx| を {SIM_SAFE_VX_MAX:.2} に抑えます（学習域は {trained:.2}）。"
            );
            eprintln!("policy-sim: 解除・変更は --vx-max VALUE。実機側では抑えません。");
            SIM_SAFE_VX_MAX
        })
    });
    let clamp = cmd_clamp(&ctl, vx_max);
    let cmd = Arc::new(Mutex::new(clamp(a.cmd0)));
    let quit = Arc::new(AtomicBool::new(false));
    let kb = if a.keyboard {
        Some(spawn_keyboard(cmd.clone(), quit.clone(), clamp.clone())?)
    } else {
        None
    };

    ctl.reset();
    let mut odom = LegOdometry::new_with_planar_slip_calibration(a.odom_calibrated);
    if a.odom_calibrated {
        eprintln!("policy-sim: 接地脚の速度依存滑り補正を有効化しました");
    }
    let truth_vel = std::env::var("GO2_SIM_TRUTH_VEL").ok().as_deref() == Some("1");
    if truth_vel {
        eprintln!(
            "policy-sim: GO2_SIM_TRUTH_VEL=1 — 体速度入力に MuJoCo の真値を使います（診断用）"
        );
    }
    let mut held: PolicyTick = ctl.hold();
    let mut faults: u32 = 0;
    let mut fell: Option<String> = None;
    let wall_start = Instant::now();
    let mut status = Instant::now();
    // 追従の要約（planned と measured の差の最大）
    let mut track_err_max = 0.0f64;
    let mut vx_sum = 0.0f64;
    let mut vx_n = 0u64;
    let mut vx_truth_sum = 0.0f64;
    let mut candidate_sum = [[0.0f64; 3]; 4];
    let mut candidate_n = [0u64; 4];
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let mut roll_abs_max = 0.0f64;
    let mut pitch_abs_max = 0.0f64;

    let mut k: u64 = 0;
    while !quit.load(Ordering::Relaxed) {
        let t = k as f64 * CONTROL_DT;
        if let Some(d) = a.duration {
            if t >= d {
                break;
            }
        }
        let cmd_now = *cmd.lock().unwrap();

        // ── 観測 → ObsInput（実機は LowState から、ここは Observation から）──
        let imu = obs.imu.ok_or("MuJoCo が IMU を返しません")?;
        let quat = quat_from_rpy(imu.rpy_rad);
        let mut inp = ObsInput::default();
        inp.quat_wxyz = quat;
        inp.gyro_rad_s = imu.gyro_rad_s;
        inp.accel_m_s2 = imu.accel_m_s2;
        for i in 0..12 {
            let st = &obs.axes()[i];
            inp.joint_q_go2[MISA_TO_GO2[i]] = st.position_rad;
            inp.joint_dq_go2[MISA_TO_GO2[i]] = st.velocity_rad_s;
        }
        let mut q_isaac_meas = [0.0f64; 12];
        for g in 0..12 {
            q_isaac_meas[GO2_TO_ISAAC[g]] = inp.joint_q_go2[g];
        }

        // 転倒判定（シムなので即終了でよい）。
        let base = plant.base_position().unwrap_or([0.0, 0.0, body_height]);
        z_min = z_min.min(base[2]);
        z_max = z_max.max(base[2]);
        roll_abs_max = roll_abs_max.max(imu.rpy_rad[0].abs());
        pitch_abs_max = pitch_abs_max.max(imu.rpy_rad[1].abs());
        if base[2] < FALL_HEIGHT_M
            || imu.rpy_rad[0].abs() > FALL_TILT_RAD
            || imu.rpy_rad[1].abs() > FALL_TILT_RAD
        {
            fell = Some(format!(
                "t={t:.2}s で転倒（z={:.2} m, roll={:.0}°, pitch={:.0}°）",
                base[2],
                imu.rpy_rad[0].to_degrees(),
                imu.rpy_rad[1].to_degrees()
            ));
            break;
        }

        // 脚オドメトリ（実機と同じ経路）。
        // Natural は計画 swing、Pure は計画が無いので接地センサを使う。
        let stance = match ctl.swing() {
            Some(sw) => [!sw[0], !sw[1], !sw[2], !sw[3]],
            None => {
                core::array::from_fn(|l| obs.contacts.get(l).copied().flatten().unwrap_or(true))
            }
        };
        odom.update(
            &inp.quat_wxyz,
            inp.gyro_rad_s,
            &inp.joint_q_go2,
            &inp.joint_dq_go2,
            stance,
            CONTROL_DT,
        );

        // ── 50 Hz: 推論 ──
        if k % DECIMATION == 0 {
            // Pure76 は体速度推定を**入力**に使う。実機は脚オドメトリしか
            // 無いが、学習時の入力は真値（Isaac の base_lin_vel）だった。
            // GO2_SIM_TRUTH_VEL=1 で MuJoCo の真値に差し替えて、推定誤差が
            // 効いているのかを切り分けられるようにしておく。
            let vel_world = if truth_vel {
                plant
                    .sim()
                    .body_world_linear_velocity("base")
                    .unwrap_or(odom.vel_world())
            } else {
                odom.vel_world()
            };
            let vel_body = world_to_body(&inp.quat_wxyz, vel_world);
            match ctl.tick(&inp, cmd_now, vel_body) {
                Ok(tk) => {
                    if !tk.anomalies.is_empty() {
                        faults += 1;
                        eprintln!("policy-sim: 観測異常 {:?}\r", tk.anomalies);
                    } else {
                        faults = 0;
                    }
                    held = tk;
                }
                Err(e) => {
                    faults += 1;
                    eprintln!("policy-sim: 推論失敗（保持）: {e}\r");
                    held = ctl.hold();
                }
            }
            if faults >= 10 {
                fell = Some("異常が連続したため中断".into());
                break;
            }
        }

        // ── 毎周期: τ_ff + Impedance 指令 ──
        let base_state = misa_policy_runner::BaseState {
            quat_wxyz: inp.quat_wxyz,
            vel_world: odom.vel_world(),
            gyro_rad_s: inp.gyro_rad_s,
            height_m: odom.height_m(),
        };
        let tau_isaac = ctl.support_torque(&base_state, cmd_now, &q_isaac_meas);
        // DCMotor のトルク-速度カーブは τ_ff にだけかける。学習時（Isaac）と
        // Python 側 sim2sim は PD と τ_ff の**合計**にかけるので厳密には形が
        // 違うが、PD を自分で計算して合計をクリップし純トルク指令で送る版を
        // 実装して比べたところ、軌跡が小数 3 桁まで一致した（cmd 0.6、15 s）。
        // この速度域では合計が平坦部（±23.7 / 45.43）に収まりカーブが効かない
        // ためで、分岐を残す価値が無かったので Impedance 一本に戻している。
        for i in 0..12 {
            let isaac = misa_to_isaac(i);
            let st = &obs.axes()[i];
            let ax = cmd_out.get_mut(AxisId::new(i as u16)).unwrap();
            ax.mode = ControlMode::Impedance;
            ax.position_rad = held.q_des_isaac[isaac];
            ax.velocity_rad_s = 0.0;
            ax.kp_nm_per_rad = held.kp_isaac[isaac];
            ax.kd_nm_s_per_rad = held.kd_isaac[isaac];
            ax.torque_ff_nm = dc_motor_clip(
                tau_isaac[isaac],
                st.velocity_rad_s,
                effort_limit_isaac(isaac),
            );
        }
        plant.exchange(&cmd_out, &mut obs)?;

        // 追従誤差と速度の集計（開始 2 s 以降）。
        if t >= 2.0 {
            for i in 0..12 {
                let e = (obs.axes()[i].position_rad - held.q_des_isaac[misa_to_isaac(i)]).abs();
                track_err_max = track_err_max.max(e);
            }
            vx_sum += odom.vel_world()[0];
            vx_truth_sum += plant
                .sim()
                .body_world_linear_velocity("base")
                .map(|v| v[0])
                .unwrap_or(0.0);
            vx_n += 1;
            for (leg, candidate) in odom.candidates_world().iter().enumerate() {
                if let Some(v) = candidate {
                    for axis in 0..3 {
                        candidate_sum[leg][axis] += v[axis];
                    }
                    candidate_n[leg] += 1;
                }
            }
        }

        // ── viz: planned（指令）と measured（MuJoCo 実測）を対で流す ──
        if let Some(p) = publisher.as_mut() {
            let mut planned = JointVec::zeros();
            let mut measured = JointVec::zeros();
            for i in 0..12 {
                let (leg, j) = (i / 3, i % 3);
                // Torque モードでは position_rad を載せないので、方策の
                // 目標そのものを planned として流す。
                planned.legs[leg][j] = held.q_des_isaac[misa_to_isaac(i)];
                measured.legs[leg][j] = obs.axes()[i].position_rad;
            }
            let stance_now: [bool; 4] =
                core::array::from_fn(|l| obs.contacts.get(l).copied().flatten().unwrap_or(false));
            let view = viz::BodyView {
                xy: [base[0], base[1]],
                yaw: imu.rpy_rad[2],
                z: base[2],
                rp: [0.0, 0.0],
                stance: stance_now,
            };
            let measured_view = viz::BodyView {
                rp: [imu.rpy_rad[0], imu.rpy_rad[1]],
                ..view
            };
            p.maybe_publish(|seq| {
                viz::Frames::both(
                    viz::frame(seq, t, &planned, &view),
                    viz::frame(seq, t, &measured, &measured_view),
                )
            });
        }

        // 実時間に合わせる（テレオペと見た目のため。--no-keyboard でも
        // viz が無ければ待たずに回してよいが、挙動を揃えて常に待つ）。
        if a.keyboard || a.viz {
            let next = wall_start + Duration::from_secs_f64(t + CONTROL_DT);
            if let Some(d) = next.checked_duration_since(Instant::now()) {
                std::thread::sleep(d);
            }
        }

        if status.elapsed().as_secs_f64() > 0.5 {
            eprint!(
                "\rpolicy-sim: t={t:6.1}s cmd=({:+.2},{:+.2},{:+.2}) v̂x={:+.2}m/s xy=({:+.2},{:+.2})m z={:.3}m rp=({:+4.1}°,{:+4.1}°)   ",
                cmd_now[0],
                cmd_now[1],
                cmd_now[2],
                odom.vel_world()[0],
                base[0],
                base[1],
                base[2],
                imu.rpy_rad[0].to_degrees(),
                imu.rpy_rad[1].to_degrees(),
            );
            use std::io::Write as _;
            let _ = std::io::stderr().flush();
            status = Instant::now();
        }
        k += 1;
    }
    quit.store(true, Ordering::Relaxed);
    if let Some(h) = kb {
        let _ = h.join();
    }

    let base = plant.base_position().unwrap_or([0.0; 3]);
    eprintln!(
        "\npolicy-sim: {:.1} s / 変位 ({:+.2}, {:+.2}) m / 最終高さ {:.3} m / \
         追従誤差 max {:.3} rad / v̄x(t≥2s) 真値 {:+.3} / オドメトリ {:+.3} m/s",
        k as f64 * CONTROL_DT,
        base[0],
        base[1],
        base[2],
        track_err_max,
        if vx_n > 0 {
            vx_truth_sum / vx_n as f64
        } else {
            0.0
        },
        if vx_n > 0 { vx_sum / vx_n as f64 } else { 0.0 },
    );
    eprintln!(
        "policy-sim: 姿勢範囲 |roll|max {:.2}° / |pitch|max {:.2}° / z [{:.3}, {:.3}] m",
        roll_abs_max.to_degrees(),
        pitch_abs_max.to_degrees(),
        z_min,
        z_max,
    );
    if vx_n > 0 {
        for (leg, name) in ["FL", "FR", "RL", "RR"].iter().enumerate() {
            let n = candidate_n[leg];
            if n > 0 {
                eprintln!(
                    "policy-sim: odom candidate {name}: contact {:.1}% mean=({:+.3},{:+.3},{:+.3}) m/s",
                    100.0 * n as f64 / vx_n as f64,
                    candidate_sum[leg][0] / n as f64,
                    candidate_sum[leg][1] / n as f64,
                    candidate_sum[leg][2] / n as f64,
                );
            }
        }
    }
    match fell {
        Some(reason) => Err(format!("policy-sim: {reason}")),
        None => Ok(()),
    }
}
