//! `go2-fake-robot` — **DDS の線の上で Go2 の役をする**（実機なしで手順 1〜3 を予行する）。
//!
//! ```text
//!   端末 A: ./target/release/go2-fake-robot                 # ロボットの役
//!   端末 B: ./target/release/go2-run --robot robots/go2.toml bridge
//!   端末 B: ./target/release/go2-run policy --model P.onnx --hold
//! ```
//!
//! **なぜ要るか**（go2_rl §119 E）: 実機の立ち上げ手順のうち、手順 4・5（吊り）は
//! `--suspend` で sim に予行してある。だが**手順 1・2（`check` / `bridge` = DDS の
//! 線そのもの）は予行できていなかった**。初接続の日に `rt/lowstate` が届かなかった
//! とき、原因が「配線・iface・sport_mode」なのか「こちらのコード」なのかを
//! **その場で切り分けられない**。この役をあらかじめ通しておけば、残る未知は
//! 線の側だけになる。
//!
//! **これは実機の模型ではない。** 関節は `q_des` を 1 次遅れで追うだけで、
//! 重力も接触も剛体も入っていない。**ここで出る数字は実機について何も言わない。**
//! 確かめられるのはこちら側だけ:
//!
//! | 確かめられること | どの手順 |
//! |---|---|
//! | participant / topic 名 / QoS が噛み合う | 1・2 |
//! | `rt/lowstate` の受信、`待機: LowState が届きません` の出方 | 2 |
//! | `rt/lowcmd` が本当に出ている（CRC つき）、周期 | 2・3 |
//! | **12 関節の並び**（`--fingerprint` で関節ごとに違う値を流す） | 3 |
//! | IMU の符号（`--pitch` / `--roll` で傾ける） | 3 |
//! | 途切れたときの振る舞い（`--drop-after`） | 3・中止 |
//!
//! 既定姿勢へのランプ（`policy` の段 A）も通る: 伏せ姿勢から始まるので
//! 「実測姿勢 → 既定姿勢へ 3 s」が実際に動く。

use std::f64::consts::PI;
use std::time::{Duration, Instant};

use unitree_go2::{
    crc32_core, joint, topics, LowCmd, LowState, Participant, Reader, ReaderQos, Writer, WriterQos,
    POS_STOP_F,
};

/// 電源を入れた直後の Go2 は脚を畳んで伏せている。ここから始めると
/// `policy` の段 A（実測姿勢 → 既定姿勢へ 3 s ランプ）が実際に動く。
/// Go2 SDK 順（FR, FL, RR, RL × hip/thigh/calf）。
const FOLDED: [f64; 12] = [
    0.0, 1.22, -2.72, // FR
    0.0, 1.22, -2.72, // FL
    0.0, 1.22, -2.72, // RR
    0.0, 1.22, -2.72, // RL
];

const G: f64 = 9.81;

struct Args {
    iface: Option<String>,
    rate_hz: f64,
    duration_s: f64,
    pitch_deg: f64,
    roll_deg: f64,
    follow_tc_s: f64,
    fingerprint: bool,
    freeze: bool,
    drop_after_s: f64,
    bad_crc: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            iface: std::env::var("GO2_IFACE").ok(),
            rate_hz: 500.0,
            duration_s: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            follow_tc_s: 0.02,
            fingerprint: false,
            freeze: false,
            drop_after_s: 0.0,
            bad_crc: false,
        }
    }
}

const USAGE: &str = "\
go2-fake-robot — DDS の線の上で Go2 の役をする（実機なしで手順 1〜3 を予行）

  --iface <name>      機体へ届く NIC（既定 $GO2_IFACE、無ければ DDS の既定経路）
  --rate <Hz>         lowstate の発行周期（既定 500）
  --duration <s>      これだけ経ったら終わる（既定 0 = 無期限）
  --pitch <deg>       胴体を前後に傾ける（IMU の符号を見る。+ = 鼻下げ）
  --roll <deg>        胴体を左右に傾ける（+ = 右下げ）
  --follow-tc <s>     関節が q_des を追う 1 次遅れの時定数（既定 0.02）
  --fingerprint       lowcmd を無視し、関節 g に q = 0.01*(g+1) を流す
                      （**12 関節の並びの検査**。FR(0.01,0.02,0.03) FL(0.04..) と出るはず）
  --freeze            lowcmd を無視して伏せ姿勢を流し続ける
  --drop-after <s>    これだけ経ったら**発行をやめる**（途切れたときの振る舞いを見る）
  --bad-crc           lowstate の CRC を壊して流す（受け側が見ているかの確認）
  -h, --help          これ

**実機の模型ではない。** ここで出る数字は実機について何も言わない。
確かめられるのはこちら側（topic 名・QoS・12 関節の並び・IMU の符号・診断の出方）。
";

fn parse() -> Result<Args, String> {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let need = |i: usize, argv: &[String], k: &str| -> Result<String, String> {
        argv.get(i + 1)
            .cloned()
            .ok_or_else(|| format!("{k} に値がありません"))
    };
    while i < argv.len() {
        let k = argv[i].as_str();
        match k {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--iface" => {
                a.iface = Some(need(i, &argv, k)?);
                i += 2;
            }
            "--rate" => {
                a.rate_hz = need(i, &argv, k)?.parse().map_err(|e| format!("--rate: {e}"))?;
                i += 2;
            }
            "--duration" => {
                a.duration_s = need(i, &argv, k)?.parse().map_err(|e| format!("--duration: {e}"))?;
                i += 2;
            }
            "--pitch" => {
                a.pitch_deg = need(i, &argv, k)?.parse().map_err(|e| format!("--pitch: {e}"))?;
                i += 2;
            }
            "--roll" => {
                a.roll_deg = need(i, &argv, k)?.parse().map_err(|e| format!("--roll: {e}"))?;
                i += 2;
            }
            "--follow-tc" => {
                a.follow_tc_s = need(i, &argv, k)?.parse().map_err(|e| format!("--follow-tc: {e}"))?;
                i += 2;
            }
            "--drop-after" => {
                a.drop_after_s = need(i, &argv, k)?.parse().map_err(|e| format!("--drop-after: {e}"))?;
                i += 2;
            }
            "--fingerprint" => {
                a.fingerprint = true;
                i += 1;
            }
            "--freeze" => {
                a.freeze = true;
                i += 1;
            }
            "--bad-crc" => {
                a.bad_crc = true;
                i += 1;
            }
            other => return Err(format!("知らないオプション `{other}`\n\n{USAGE}")),
        }
    }
    if a.rate_hz <= 0.0 {
        return Err("--rate は正の値".into());
    }
    if a.follow_tc_s < 0.0 {
        return Err("--follow-tc は 0 以上".into());
    }
    Ok(a)
}

/// rpy（ZYX、yaw は 0 固定）→ クォータニオン (w, x, y, z)。
fn quat_from_rp(roll: f64, pitch: f64) -> [f64; 4] {
    let (sr, cr) = (roll * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    // yaw = 0 なので (cy, sy) = (1, 0)。
    [cr * cp, sr * cp, cr * sp, -sr * sp]
}

/// 世界の重力 (0, 0, -g) を胴体系で見た**比力**（静止している加速度計の読み）。
/// R(roll, pitch) の転置を (0, 0, +g) にかける。
fn accel_body(roll: f64, pitch: f64) -> [f64; 3] {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    [-sp * G, sr * cp * G, cr * cp * G]
}

/// LowState の CRC（機体が入れているもの。`set_crc` と同じ流儀で最後の
/// 1 ワードを除く全ワードに crc32_core をかける）。
fn set_state_crc(s: &mut LowState) {
    let len = core::mem::size_of::<LowState>() / 4;
    let words: &[u32] =
        unsafe { core::slice::from_raw_parts(s as *const LowState as *const u32, len) };
    let crc = crc32_core(&words[..len - 1]);
    s.crc = crc;
}

fn main() {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("go2-fake-robot: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&a) {
        eprintln!("go2-fake-robot: {e}");
        std::process::exit(1);
    }
}

fn run(a: &Args) -> Result<(), String> {
    let dp = Participant::new(0, a.iface.as_deref())
        .map_err(|e| format!("DDS participant: {e}"))?;
    let state_topic = dp
        .create_topic::<LowState>(topics::LOW_STATE)
        .map_err(|e| format!("state topic: {e}"))?;
    let writer: Writer<LowState> = dp
        .create_writer(&state_topic, WriterQos::low_level_default())
        .map_err(|e| format!("writer: {e}"))?;
    let cmd_topic = dp
        .create_topic::<LowCmd>(topics::LOW_CMD)
        .map_err(|e| format!("cmd topic: {e}"))?;
    let reader: Reader<LowCmd> = dp
        .create_reader(&cmd_topic, ReaderQos::low_level_default())
        .map_err(|e| format!("reader: {e}"))?;

    let roll = a.roll_deg * PI / 180.0;
    let pitch = a.pitch_deg * PI / 180.0;
    let quat = quat_from_rp(roll, pitch);
    let accel = accel_body(roll, pitch);

    eprintln!(
        "go2-fake-robot: {} Hz で rt/lowstate を流し、rt/lowcmd を読みます（iface {}）",
        a.rate_hz,
        a.iface.as_deref().unwrap_or("既定")
    );
    eprintln!(
        "  姿勢 roll {:+.1}° pitch {:+.1}° → quat(w,x,y,z) = ({:+.4}, {:+.4}, {:+.4}, {:+.4})、\
         加速度 ({:+.2}, {:+.2}, {:+.2}) m/s²",
        a.roll_deg, a.pitch_deg, quat[0], quat[1], quat[2], quat[3], accel[0], accel[1], accel[2]
    );
    if a.fingerprint {
        eprintln!(
            "  --fingerprint: 関節 g に q = 0.01*(g+1) を流します —— \
             受け側が FR(0.01,0.02,0.03) FL(0.04,0.05,0.06) RR(0.07..) RL(0.10..) と\
             出せば **12 関節の並びは正しい**"
        );
    }
    if a.freeze {
        eprintln!("  --freeze: lowcmd を無視して伏せ姿勢のまま流します");
    }
    if a.drop_after_s > 0.0 {
        eprintln!("  --drop-after {:.1} s: そこで**発行をやめます**", a.drop_after_s);
    }
    if a.bad_crc {
        eprintln!("  --bad-crc: lowstate の CRC を壊して流します");
    }
    eprintln!("  **実機の模型ではありません。** 数字は実機について何も言いません。");

    let mut q = if a.fingerprint {
        core::array::from_fn(|g| 0.01 * (g + 1) as f64)
    } else {
        FOLDED
    };
    let mut dq = [0.0f64; 12];
    let mut q_des = q;
    let mut have_cmd = false;

    let dt = 1.0 / a.rate_hz;
    let t0 = Instant::now();
    let mut tick: u64 = 0;
    let mut cmd_count: u64 = 0;
    let mut cmd_count_prev: u64 = 0;
    let mut last_kp = 0.0f32;
    let mut last_kd = 0.0f32;
    let mut next_report = t0 + Duration::from_secs(1);
    let mut dropped = false;

    loop {
        let now = Instant::now();
        let t = now.duration_since(t0).as_secs_f64();
        if a.duration_s > 0.0 && t >= a.duration_s {
            break;
        }
        if a.drop_after_s > 0.0 && !dropped && t >= a.drop_after_s {
            dropped = true;
            eprintln!(
                "go2-fake-robot: {:.1} s —— **発行をやめます**（受け側が気づくか見る）",
                t
            );
        }

        // ── lowcmd を読む（溜まっていれば最新まで） ──
        while let Some(c) = reader.poll().map_err(|e| format!("poll: {e}"))? {
            cmd_count += 1;
            if !(a.fingerprint || a.freeze) {
                for g in 0..joint::NUM_LEG_JOINTS {
                    let m = &c.motor_cmd[g];
                    // q = PosStopF は「位置制御を切る」印。保持する。
                    if m.q != POS_STOP_F && m.q.is_finite() {
                        q_des[g] = m.q as f64;
                    }
                }
            }
            last_kp = c.motor_cmd[0].kp;
            last_kd = c.motor_cmd[0].kd;
            have_cmd = true;
        }

        // ── 関節を 1 次遅れで追わせる（**これは物理ではない**） ──
        if have_cmd && !(a.fingerprint || a.freeze) {
            let alpha = if a.follow_tc_s > 0.0 {
                (dt / a.follow_tc_s).min(1.0)
            } else {
                1.0
            };
            // kp = 0（脱力）のときは追わない。実機も追わない。
            let gate = if last_kp > 0.0 { 1.0 } else { 0.0 };
            for g in 0..12 {
                let step = gate * alpha * (q_des[g] - q[g]);
                q[g] += step;
                dq[g] = step / dt;
            }
        } else {
            dq = [0.0; 12];
        }

        // ── LowState を組んで流す ──
        if !dropped {
            let mut s = LowState::default();
            s.head = [0xFE, 0xEF];
            s.level_flag = 0xFF;
            s.tick = tick as u32;
            s.imu_state.quaternion = [quat[0] as f32, quat[1] as f32, quat[2] as f32, quat[3] as f32];
            s.imu_state.rpy = [roll as f32, pitch as f32, 0.0];
            s.imu_state.gyroscope = [0.0, 0.0, 0.0];
            s.imu_state.accelerometer =
                [accel[0] as f32, accel[1] as f32, accel[2] as f32];
            s.imu_state.temperature = 30;
            for g in 0..12 {
                let m = &mut s.motor_state[g];
                m.mode = 0x01;
                m.q = q[g] as f32;
                m.dq = dq[g] as f32;
                m.q_raw = m.q;
                m.dq_raw = m.dq;
                m.temperature = 30;
            }
            // 接地判定のしきい値は 20（`go2_plant.rs` の CONTACT_THRESHOLD）。
            // 4 脚すべて接地している体にする。
            s.foot_force = [120, 120, 120, 120];
            s.foot_force_est = s.foot_force;
            s.bms_state.soc = 80;
            s.power_v = 29.0;
            s.power_a = 1.5;
            set_state_crc(&mut s);
            if a.bad_crc {
                s.crc ^= 0xFFFF_FFFF;
            }
            writer.write(&s).map_err(|e| format!("write: {e}"))?;
        }
        tick += 1;

        // ── 1 Hz の報告 ──
        if now >= next_report {
            let hz = (cmd_count - cmd_count_prev) as f64;
            let worst = (0..12)
                .map(|g| (q_des[g] - q[g]).abs())
                .fold(0.0f64, f64::max);
            eprintln!(
                "  t {:5.1} s  lowstate {} 発行{}  lowcmd {:4.0} Hz（計 {}）  \
                 kp {:5.1} kd {:4.2}  |q_des−q| max {:.3} rad",
                t,
                tick,
                if dropped { "（停止中）" } else { "" },
                hz,
                cmd_count,
                last_kp,
                last_kd,
                worst
            );
            cmd_count_prev = cmd_count;
            next_report += Duration::from_secs(1);
        }

        // ── 絶対締切まで眠る ──
        let next = t0 + Duration::from_secs_f64(dt * tick as f64);
        if let Some(d) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(d);
        }
    }
    eprintln!(
        "go2-fake-robot: 終わり（lowstate {} 発行、lowcmd {} 受信）",
        tick, cmd_count
    );
    Ok(())
}
