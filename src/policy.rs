//! `policy` サブコマンド — MIT モード RL 方策（Natural 契約）を実機で回す。
//!
//! misa-runner の `run` ループは使わない: あちらは歩容コントローラに配線が
//! 固定で、50 Hz 推論 + 500 Hz MIT 指令 + 毎周期の τ_ff という二層構造の
//! 差し込み口が無い。ここで Go2Plant を**具象型のまま**使い、自前の 500 Hz
//! ループを回す（misa-runner にコントローラの seam が入ったら移す）。
//!
//! ```text
//! go2-run policy --model exported/policy.onnx [--iface eth0]
//!                [--vx V] [--vy V] [--wz W] [--duration S]
//!                [--stride-gain 1.55] [--yaw-stride-gain 2.0] [--body-height 0.30]
//!                [--hold] [--no-keyboard] [--no-release]
//! ```
//!
//! 段取り（go2-gait-runner の policy モードで実証済みの形）:
//!   A. sport_mode 解除 → 実測姿勢から方策の既定姿勢へ 3 s ランプ（kp 0→45）
//!   B. 0.5 s 保持 → クロックをリセットして方策ループ
//!   C. q/Esc か --duration で抜け、伏せ姿勢へ 2.5 s ランプ → 脱力
//!
//! 安全側: 観測スクリーンの異常・推論失敗は直前の指令を保持（連発で中断）、
//! 傾き 35° で即脱力（立て直しより転倒完了のほうが安全）。

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use misa_policy_runner::go2::{
    clamp_cmd, dc_motor_clip, effort_limit_isaac, GO2_TO_ISAAC, ISAAC_TO_GO2,
};
use misa_policy_runner::{
    BaseState, NaturalController, ObsInput, OnnxPolicy, PolicyTick, TrajectoryCfg,
};

use crate::backend::{iface_from_env, release_sport_mode};
use crate::estimator::LegOdometry;
use crate::go2_plant::Go2Plant;

/// 500 Hz の低レベル周期。
const CONTROL_DT: f64 = 0.002;
/// 50 Hz 推論（10 tick に 1 回）。
const DECIMATION: u64 = 10;
const RAMP_SECS: f64 = 3.0;
const FOLD_SECS: f64 = 2.5;
/// 伏せ姿勢（Go2 モータ順 FR,FL,RR,RL × hip/thigh/calf。go2-gait-runner と同じ）。
const LIE_POS: [f64; 12] = [
    0.0, 1.36, -2.65, // FR
    0.0, 1.36, -2.65, // FL
    -0.2, 1.36, -2.65, // RR
    0.2, 1.36, -2.65, // RL
];
/// これを超えたら即脱力（rad）。
const TILT_ABORT_RAD: f64 = 35.0 * std::f64::consts::PI / 180.0;
/// 推論失敗・観測異常がこの回数続いたら中断。
const MAX_CONSECUTIVE_FAULTS: u32 = 10;

struct Args {
    model: String,
    iface: Option<String>,
    cmd0: [f64; 3],
    duration: Option<f64>,
    cfg: TrajectoryCfg,
    hold: bool,
    keyboard: bool,
    release: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut out = Args {
        model: String::new(),
        iface: iface_from_env(),
        cmd0: [0.0; 3],
        duration: None,
        cfg: TrajectoryCfg::h30_standard(),
        hold: false,
        keyboard: true,
        release: true,
    };
    fn val(it: &mut std::slice::Iter<'_, String>, name: &str) -> Result<f64, String> {
        it.next()
            .ok_or(format!("{name} に値がありません"))?
            .parse()
            .map_err(|e| format!("{name}: {e}"))
    }
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => out.model = it.next().ok_or("--model に値がありません")?.clone(),
            "--iface" => out.iface = Some(it.next().ok_or("--iface に値がありません")?.clone()),
            "--vx" => out.cmd0[0] = val(&mut it, "--vx")?,
            "--vy" => out.cmd0[1] = val(&mut it, "--vy")?,
            "--wz" => out.cmd0[2] = val(&mut it, "--wz")?,
            "--duration" => out.duration = Some(val(&mut it, "--duration")?),
            "--stride-gain" => out.cfg.stride_gain = val(&mut it, "--stride-gain")?,
            "--yaw-stride-gain" => {
                out.cfg.yaw_stride_gain = Some(val(&mut it, "--yaw-stride-gain")?)
            }
            "--body-height" => out.cfg.body_height = Some(val(&mut it, "--body-height")?),
            "--hold" => out.hold = true,
            "--no-keyboard" => out.keyboard = false,
            "--no-release" => out.release = false,
            other => return Err(format!("policy: 知らないオプション {other:?}")),
        }
    }
    if out.model.is_empty() {
        return Err("--model PATH は必須です（exported/policy.onnx、39 入力）".into());
    }
    Ok(out)
}

/// WASD テレオペ。指令は共有 cmd を書き換え、q/Esc/Ctrl-C で quit。
fn spawn_keyboard(
    cmd: Arc<Mutex<[f64; 3]>>,
    quit: Arc<AtomicBool>,
) -> Result<std::thread::JoinHandle<()>, String> {
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
    enable_raw_mode().map_err(|e| format!("raw mode: {e}"))?;
    eprintln!(
        "policy: keys — W/S = vx±0.02, A/D = wz±0.05, R/F = vy±0.02,\r\n\
         \x20      Space = 全部 0, Esc / q = 終了（伏せて抜ける）\r"
    );
    Ok(std::thread::spawn(move || {
        loop {
            if quit.load(Ordering::Relaxed) {
                break;
            }
            if let Ok(true) = event::poll(Duration::from_millis(100)) {
                if let Ok(Event::Key(k)) = event::read() {
                    let mut c = cmd.lock().unwrap();
                    match k.code {
                        KeyCode::Char('w') | KeyCode::Up => c[0] += 0.02,
                        KeyCode::Char('s') | KeyCode::Down => c[0] -= 0.02,
                        KeyCode::Char('a') | KeyCode::Left => c[2] += 0.05,
                        KeyCode::Char('d') | KeyCode::Right => c[2] -= 0.05,
                        KeyCode::Char('r') => c[1] += 0.02,
                        KeyCode::Char('f') => c[1] -= 0.02,
                        KeyCode::Char(' ') => *c = [0.0; 3],
                        KeyCode::Char('q') | KeyCode::Esc => {
                            quit.store(true, Ordering::Relaxed)
                        }
                        KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                            quit.store(true, Ordering::Relaxed)
                        }
                        _ => {}
                    }
                    *c = clamp_cmd(*c);
                }
            }
        }
        let _ = disable_raw_mode();
    }))
}

fn obs_input_from(s: &unitree_go2::LowState) -> ObsInput {
    let mut inp = ObsInput::default();
    let im = &s.imu_state;
    inp.quat_wxyz = [
        im.quaternion[0] as f64,
        im.quaternion[1] as f64,
        im.quaternion[2] as f64,
        im.quaternion[3] as f64,
    ];
    inp.gyro_rad_s = [
        im.gyroscope[0] as f64,
        im.gyroscope[1] as f64,
        im.gyroscope[2] as f64,
    ];
    inp.accel_m_s2 = [
        im.accelerometer[0] as f64,
        im.accelerometer[1] as f64,
        im.accelerometer[2] as f64,
    ];
    for g in 0..12 {
        inp.joint_q_go2[g] = s.motor_state[g].q as f64;
        inp.joint_dq_go2[g] = s.motor_state[g].dq as f64;
    }
    inp
}

pub fn run(args: &[String]) -> Result<(), String> {
    let a = parse(args)?;

    let policy = OnnxPolicy::load(&a.model, 39)?;
    let mut ctl = NaturalController::new(policy, a.cfg)?;
    eprintln!(
        "policy: {} を読み込みました（stride {:.2}/{:.2}, height {:.2} m）",
        a.model,
        a.cfg.stride_gain,
        a.cfg.yaw_stride_gain.unwrap_or(a.cfg.stride_gain),
        a.cfg.body_height.unwrap_or(f64::NAN),
    );

    if a.release {
        release_sport_mode(a.iface.as_deref())?;
    }
    // Position モードのゲインは policy モードでは使わない（send_raw 直行）。
    let mut plant = Go2Plant::connect(a.iface.as_deref(), [45.0; 3], [2.0; 3])?;

    // ── A: 実測姿勢 → 方策の既定姿勢（ik(trajectory(0,0))）へランプ ──
    let start: [f64; 12] = {
        let s = plant.last_state().ok_or("LowState がありません")?;
        core::array::from_fn(|j| s.motor_state[j].q as f64)
    };
    let mut default_go2 = [0.0f64; 12];
    for i in 0..12 {
        default_go2[ISAAC_TO_GO2[i]] = ctl.default_pose_isaac()[i];
    }
    let (kp0, kd0) = ctl.initial_gains();

    let loop_start = Instant::now();
    let mut tick: u64 = 0;
    let zeros = [0.0f64; 12];
    /// 1 フレーム送って絶対締切まで眠る（一様 kp/kd 用: ランプと伏せ）。
    fn emit(
        plant: &mut Go2Plant,
        tick: &mut u64,
        loop_start: Instant,
        q: &[f64; 12],
        kp: f64,
        kd: f64,
    ) -> Result<(), String> {
        let zeros = [0.0f64; 12];
        plant.send_raw(q, &zeros, &[kp; 12], &[kd; 12], &zeros)?;
        *tick += 1;
        let next = loop_start + Duration::from_secs_f64(CONTROL_DT * *tick as f64);
        if let Some(d) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(d);
        }
        Ok(())
    }

    eprintln!("policy: 既定姿勢へ {RAMP_SECS} s でランプします（kp 0→{kp0}）");
    let ramp_n = (RAMP_SECS / CONTROL_DT) as u64;
    for i in 0..ramp_n {
        let p = i as f64 / ramp_n as f64;
        let q: [f64; 12] = core::array::from_fn(|j| (1.0 - p) * start[j] + p * default_go2[j]);
        emit(&mut plant, &mut tick, loop_start, &q, kp0 * p, kd0)?;
        plant.poll_state()?;
    }
    for _ in 0..(0.5 / CONTROL_DT) as u64 {
        emit(&mut plant, &mut tick, loop_start, &default_go2, kp0, kd0)?;
        plant.poll_state()?;
    }

    // ── B: 方策ループ ──
    let cmd = Arc::new(Mutex::new(clamp_cmd(a.cmd0)));
    let quit = Arc::new(AtomicBool::new(false));
    let kb = if a.keyboard {
        Some(spawn_keyboard(cmd.clone(), quit.clone())?)
    } else {
        eprintln!(
            "policy: キーボード無効。vx={:.2} vy={:.2} wz={:.2} を保持",
            a.cmd0[0], a.cmd0[1], a.cmd0[2]
        );
        None
    };

    ctl.reset();
    let mut odom = LegOdometry::new();
    let mut held: PolicyTick = ctl.hold();
    let mut faults: u32 = 0;
    let mut status = Instant::now();
    let run_start = Instant::now();
    let mut abort: Option<String> = None;
    if a.hold {
        eprintln!("policy: --hold — 方策は走らせず既定姿勢を保持して観測だけ表示します\r");
    } else {
        eprintln!("policy: RUNNING\r");
    }

    let mut k: u64 = 0;
    'run: while !quit.load(Ordering::Relaxed) {
        if let Some(d) = a.duration {
            if run_start.elapsed().as_secs_f64() >= d {
                break;
            }
        }
        plant.poll_state()?;
        let cmd_now = *cmd.lock().unwrap();
        let (inp, q_isaac_meas, dq_go2) = {
            let s = plant.last_state().ok_or("LowState がありません")?;
            let inp = obs_input_from(s);
            let mut qi = [0.0f64; 12];
            for g in 0..12 {
                qi[GO2_TO_ISAAC[g]] = inp.joint_q_go2[g];
            }
            (inp, qi, inp.joint_dq_go2)
        };

        // 観測が腐ったら（バス断など）保持のまま数えて中断へ。
        if plant.state_age() > Duration::from_millis(20) {
            faults += 1;
        }

        // 傾きの即時脱力（projected gravity の z から）。
        let g_b = misa_policy_runner::support::projected_gravity(&inp.quat_wxyz);
        let tilt = (-g_b[2]).clamp(-1.0, 1.0).acos();
        if tilt > TILT_ABORT_RAD {
            plant.send_limp()?;
            abort = Some(format!(
                "傾き {:.0}° — 脱力して中断しました",
                tilt.to_degrees()
            ));
            break 'run;
        }

        // 脚オドメトリ（接地 = 方策側の計画 swing の否定）。
        let stance = {
            let sw = ctl.swing();
            [!sw[0], !sw[1], !sw[2], !sw[3]]
        };
        odom.update(
            &inp.quat_wxyz,
            inp.gyro_rad_s,
            &inp.joint_q_go2,
            &dq_go2,
            stance,
            CONTROL_DT,
        );

        // 50 Hz: 推論。失敗は直前の指令を保持。
        if !a.hold && k % DECIMATION == 0 {
            match ctl.tick(&inp, cmd_now) {
                Ok(t) => {
                    if t.anomalies.is_empty() {
                        faults = 0;
                    } else {
                        faults += 1;
                        eprintln!("policy: 観測異常 {:?}\r", t.anomalies);
                    }
                    held = t;
                }
                Err(e) => {
                    faults += 1;
                    eprintln!("policy: 推論失敗（保持）: {e}\r");
                    held = ctl.hold();
                }
            }
            if faults >= MAX_CONSECUTIVE_FAULTS {
                abort = Some("異常が連続したため中断しました".into());
                break 'run;
            }
        }

        // 毎周期: 支持脚レンチ τ_ff（--hold では 0）。
        let base = BaseState {
            quat_wxyz: inp.quat_wxyz,
            vel_world: odom.vel_world(),
            gyro_rad_s: inp.gyro_rad_s,
            height_m: odom.height_m(),
        };
        let tau_isaac = if a.hold {
            [0.0; 12]
        } else {
            ctl.support_torque(&base, cmd_now, &q_isaac_meas)
        };

        // Isaac → Go2 順へ。τ_ff は学習時のトルク包絡で二重クランプ。
        let mut q_go2 = [0.0f64; 12];
        let mut kp_go2 = [0.0f64; 12];
        let mut kd_go2 = [0.0f64; 12];
        let mut tau_go2 = [0.0f64; 12];
        let hold_tick;
        let src = if a.hold {
            hold_tick = ctl.hold();
            &hold_tick
        } else {
            &held
        };
        for i in 0..12 {
            let g = ISAAC_TO_GO2[i];
            q_go2[g] = src.q_des_isaac[i];
            kp_go2[g] = src.kp_isaac[i];
            kd_go2[g] = src.kd_isaac[i];
            let dq_i = dq_go2[g];
            tau_go2[g] = dc_motor_clip(tau_isaac[i], dq_i, effort_limit_isaac(i));
        }
        plant.send_raw(&q_go2, &zeros, &kp_go2, &kd_go2, &tau_go2)?;
        tick += 1;
        let next = loop_start + Duration::from_secs_f64(CONTROL_DT * tick as f64);
        if let Some(d) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(d);
        }

        if status.elapsed().as_secs_f64() > 0.5 {
            eprint!(
                "\rpolicy: t={:6.1}s cmd=({:+.2},{:+.2},{:+.2}) v=({:+.2},{:+.2}) h={:.2}m tilt={:4.1}°   ",
                ctl.gait_time_s(),
                cmd_now[0],
                cmd_now[1],
                cmd_now[2],
                base.vel_world[0],
                base.vel_world[1],
                base.height_m,
                tilt.to_degrees(),
            );
            let _ = std::io::stderr().flush();
            status = Instant::now();
        }
        k += 1;
    }
    quit.store(true, Ordering::Relaxed);

    // ── C: 伏せへランプして脱力 ──
    if abort.is_none() {
        eprintln!("\npolicy: 伏せ姿勢へ {FOLD_SECS} s でランプします");
        let cur: [f64; 12] = {
            let s = plant.last_state().ok_or("LowState がありません")?;
            core::array::from_fn(|j| s.motor_state[j].q as f64)
        };
        let fold_n = (FOLD_SECS / CONTROL_DT) as u64;
        for i in 0..fold_n {
            let p = i as f64 / fold_n as f64;
            let q: [f64; 12] = core::array::from_fn(|j| (1.0 - p) * cur[j] + p * LIE_POS[j]);
            emit(&mut plant, &mut tick, loop_start, &q, 40.0, 2.0)?;
            plant.poll_state()?;
        }
        for _ in 0..(0.5 / CONTROL_DT) as u64 {
            emit(&mut plant, &mut tick, loop_start, &LIE_POS, 40.0, 2.0)?;
        }
        plant.send_limp()?;
        eprintln!("policy: 伏せて脱力しました");
    }

    if let Some(h) = kb {
        let _ = h.join();
    }
    match abort {
        Some(reason) => Err(format!("policy: {reason}")),
        None => Ok(()),
    }
}
