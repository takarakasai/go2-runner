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
use crate::policy::{spawn_keyboard, world_to_body, Args, Ctl};

const CONTROL_DT: f64 = 0.002;
const DECIMATION: u64 = 10;
/// 転倒判定: 高さがここを割るか、roll/pitch がここを超えたら止める。
const FALL_HEIGHT_M: f64 = 0.12;
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

pub(crate) fn run(a: &Args) -> Result<(), String> {
    let mut ctl = Ctl::load(a)?;
    // Pure 契約は 0.30 m のしゃがみ姿勢で学習されている（doc/mit_pure.md）。
    let body_height = a
        .cfg
        .body_height
        .unwrap_or(if matches!(ctl, Ctl::Pure(_)) { 0.30 } else { 0.40 });
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
    let opts = SimOptions {
        misa_path: a.misa.clone(),
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
        friction: a.friction.map(|mu| [mu, 0.005, 0.0001]),
        impratio: None,
        cone: None,
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

    let clamp = ctl.clamp_fn();
    let cmd = Arc::new(Mutex::new(clamp(a.cmd0)));
    let quit = Arc::new(AtomicBool::new(false));
    let kb = if a.keyboard {
        Some(spawn_keyboard(cmd.clone(), quit.clone(), clamp)?)
    } else {
        None
    };

    ctl.reset();
    let mut odom = LegOdometry::new();
    let truth_vel = std::env::var("GO2_SIM_TRUTH_VEL").ok().as_deref() == Some("1");
    if truth_vel {
        eprintln!("policy-sim: GO2_SIM_TRUTH_VEL=1 — 体速度入力に MuJoCo の真値を使います（診断用）");
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
            None => core::array::from_fn(|l| {
                obs.contacts.get(l).copied().flatten().unwrap_or(true)
            }),
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
        for i in 0..12 {
            let isaac = misa_to_isaac(i);
            let ax = cmd_out.get_mut(AxisId::new(i as u16)).unwrap();
            ax.mode = ControlMode::Impedance;
            ax.position_rad = held.q_des_isaac[isaac];
            ax.velocity_rad_s = 0.0;
            ax.kp_nm_per_rad = held.kp_isaac[isaac];
            ax.kd_nm_s_per_rad = held.kd_isaac[isaac];
            ax.torque_ff_nm = dc_motor_clip(
                tau_isaac[isaac],
                obs.axes()[i].velocity_rad_s,
                effort_limit_isaac(isaac),
            );
        }
        plant.exchange(&cmd_out, &mut obs)?;

        // 追従誤差と速度の集計（開始 2 s 以降）。
        if t >= 2.0 {
            for i in 0..12 {
                let e = (obs.axes()[i].position_rad - cmd_out.get(AxisId::new(i as u16)).unwrap().position_rad).abs();
                track_err_max = track_err_max.max(e);
            }
            vx_sum += odom.vel_world()[0];
            vx_truth_sum += plant
                .sim()
                .body_world_linear_velocity("base")
                .map(|v| v[0])
                .unwrap_or(0.0);
            vx_n += 1;
        }

        // ── viz: planned（指令）と measured（MuJoCo 実測）を対で流す ──
        if let Some(p) = publisher.as_mut() {
            let mut planned = JointVec::zeros();
            let mut measured = JointVec::zeros();
            for i in 0..12 {
                let (leg, j) = (i / 3, i % 3);
                planned.legs[leg][j] = cmd_out.get(AxisId::new(i as u16)).unwrap().position_rad;
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
        if vx_n > 0 { vx_truth_sum / vx_n as f64 } else { 0.0 },
        if vx_n > 0 { vx_sum / vx_n as f64 } else { 0.0 },
    );
    match fell {
        Some(reason) => Err(format!("policy-sim: {reason}")),
        None => Ok(()),
    }
}
