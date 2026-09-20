//! `policy` サブコマンド — MIT モード RL 方策（Natural 契約）を実機で回す。
//!
//! misa-runner の `run` ループは使わない: あちらは歩容コントローラに配線が
//! 固定で、50 Hz 推論 + 500 Hz MIT 指令 + 毎周期の τ_ff という二層構造の
//! 差し込み口が無い。ここで Go2Plant を**具象型のまま**使い、自前の 500 Hz
//! ループを回す（misa-runner にコントローラの seam が入ったら移す）。
//!
//! ```text
//! go2-run policy --model exported/policy.onnx [--estimator estimator.onnx]
//!                [--iface eth0]
//!                [--vx V] [--vy V] [--wz W] [--duration S]
//!                [--stride-gain 1.55] [--yaw-stride-gain 2.0]
//!                [--body-height 0.30 | --low-stance]
//!                [--vx-max 0.6] [--joint-damping 0.2] [--hold]
//!                [--no-keyboard] [--no-release]
//! ```
//!
//! 立位高さ: `--body-height` は旋回ストライドゲインを自動で較正し直す
//! (`TrajectoryCfg::with_height`)。低くすると同じ方策が旋回を過追従する
//! ため（0.21 m で 133%）で、明示の `--yaw-stride-gain` があればそちらが
//! 優先される。`--low-stance` は推奨の 0.22 m / 1.644 の別名。
//! 同じ ONNX のまま立位の押し耐性が 11% → 60%、耐力が体重比 0.37 →
//! 0.54–0.80 に上がり、学習範囲内 (|vx| ≤ 0.16 m/s) の追従は同等以上。
//! 代償は腹下クリアランスが 8 cm 減ること。段差を越える場面では 0.30 m
//! に戻す（go2_rl `doc/push_robustness.md` §7）。
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
use misa_policy_runner::gru::{HistoryVelocityEstimator, VelocitySource};
use misa_policy_runner::{
    clamp_gru_cmd, clamp_pure_cmd, graph_input_arity, BaseState, NaturalController, ObsInput,
    OnnxPolicy, PolicyTick, PureController, PureGruController, RecurrentOnnxPolicy, TrajectoryCfg,
};

use crate::backend::{iface_from_env, release_sport_mode};
use crate::estimator::LegOdometry;
use crate::go2_plant::{Go2Plant, CONTACT_THRESHOLD, MISA_LEG_TO_GO2_FOOT};

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

pub(crate) struct Args {
    pub model: String,
    pub iface: Option<String>,
    pub cmd0: [f64; 3],
    pub duration: Option<f64>,
    pub cfg: TrajectoryCfg,
    pub hold: bool,
    pub keyboard: bool,
    pub release: bool,
    /// 実機ではなく MuJoCo で回す（--features sim のビルド）。
    pub sim: bool,
    /// --sim: 読み込む .misa（既定 models/unitree_go2/go2.misa）。
    pub misa: String,
    /// --sim: articara へのライブ配信（Zenoh）。
    pub viz: bool,
    pub viz_endpoint: Option<String>,
    pub viz_rate_hz: f64,
    /// --sim: 接地摩擦の滑り成分（既定 articara の 0.7）。
    pub friction: Option<f64>,
    /// |vx| の追加上限（契約のクランプの後にかける安全弁）。テレオペで
    /// プラントが支えられない速度まで上げてしまうのを防ぐ。
    pub vx_max: Option<f64>,
    /// --sim: MuJoCo の `<option impratio>`（既定 100 = Python 参照プラントと同じ）。
    pub impratio: Option<f64>,
    /// --sim: MuJoCo の `<option cone>`（既定 elliptic、同上）。
    pub cone: Option<String>,
    /// --sim: 関節の受動粘性（N·m·s/rad）を .misa の値から差し替える。
    /// go2.misa は MuJoCo Menagerie 由来の 2.0 で、数値安定性向けの
    /// 保守的な値。実機はこれよりずっと小さい（詳細は README）。
    pub joint_damping: Option<f64>,
    /// Apply the measured contact-foot slip curve to Pure76 velocity input.
    pub odom_calibrated: bool,
    /// GRU 契約の凍結速度推定器（6×73 → 3）。再帰グラフには必須で、
    /// `--gru-host-velocity` を付けたときだけ省ける。
    pub estimator: Option<String>,
    /// GRU 契約の速度入力を推定器ではなく脚オドメトリにする（診断用）。
    /// 契約から外れる — 別の入力分布になる（doc/gru_deploy.md）。
    pub gru_host_velocity: bool,
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
        sim: false,
        misa: "models/unitree_go2/go2.misa".into(),
        viz: false,
        viz_endpoint: None,
        viz_rate_hz: 100.0,
        friction: None,
        vx_max: None,
        impratio: None,
        cone: None,
        joint_damping: None,
        odom_calibrated: false,
        estimator: None,
        gru_host_velocity: false,
    };
    fn val(it: &mut std::slice::Iter<'_, String>, name: &str) -> Result<f64, String> {
        it.next()
            .ok_or(format!("{name} に値がありません"))?
            .parse()
            .map_err(|e| format!("{name}: {e}"))
    }
    // --body-height recalibrates the yaw stride gain (misa_policy_runner
    // TrajectoryCfg::with_height). An explicit --yaw-stride-gain wins,
    // whatever the flag order.
    let mut yaw_gain_pinned = false;
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
                out.cfg.yaw_stride_gain = Some(val(&mut it, "--yaw-stride-gain")?);
                yaw_gain_pinned = true;
            }
            "--body-height" => {
                let h = val(&mut it, "--body-height")?;
                let pinned = out.cfg.yaw_stride_gain;
                out.cfg = out.cfg.with_height(h);
                if yaw_gain_pinned {
                    out.cfg.yaw_stride_gain = pinned;
                }
            }
            "--low-stance" => {
                let pinned = out.cfg.yaw_stride_gain;
                out.cfg = TrajectoryCfg::h30_low_stance();
                if yaw_gain_pinned {
                    out.cfg.yaw_stride_gain = pinned;
                }
            }
            "--hold" => out.hold = true,
            "--no-keyboard" => out.keyboard = false,
            "--no-release" => out.release = false,
            "--sim" => out.sim = true,
            "--misa" => out.misa = it.next().ok_or("--misa に値がありません")?.clone(),
            "--viz" => out.viz = true,
            "--viz-endpoint" => {
                out.viz_endpoint = Some(it.next().ok_or("--viz-endpoint に値がありません")?.clone())
            }
            "--viz-rate" => out.viz_rate_hz = val(&mut it, "--viz-rate")?,
            "--friction" => out.friction = Some(val(&mut it, "--friction")?),
            "--vx-max" => out.vx_max = Some(val(&mut it, "--vx-max")?),
            "--joint-damping" => out.joint_damping = Some(val(&mut it, "--joint-damping")?),
            "--odom-calibrated" => out.odom_calibrated = true,
            "--estimator" => {
                out.estimator = Some(it.next().ok_or("--estimator に値がありません")?.clone())
            }
            "--gru-host-velocity" => out.gru_host_velocity = true,
            "--impratio" => out.impratio = Some(val(&mut it, "--impratio")?),
            "--cone" => out.cone = Some(it.next().ok_or("--cone に値がありません")?.clone()),
            other => return Err(format!("policy: 知らないオプション {other:?}")),
        }
    }
    if out.model.is_empty() {
        return Err(
            "--model PATH は必須です（exported/policy.onnx。入力が 1 本なら幅で \
             39 = Natural / 73・76 = Pure、2 本なら GRU 契約（76 + 隠れ状態 128）。\
             GRU には --estimator も要ります）"
                .into(),
        );
    }
    Ok(out)
}

/// 契約ディスパッチ — まずグラフの**入力の本数**、次に幅で判別する。
///
/// 入力 1 本: 39 = Natural（リファレンス + 残差 + τ_ff）、73/76 = Pure
/// （ネットワークのみ、76 は体速度推定を追加入力）。
/// 入力 2 本: GRU（76 + 隠れ状態 128 の再帰方策）。
///
/// **幅だけで Pure76 と GRU を見分けてはいけない** — どちらも 76 入力で、
/// 中身（隠れ状態・速度の出どころ・指令域）がまるごと違う
/// （go2_rl doc/gru_runtime_contract_audit_20260920.md）。
pub(crate) enum Ctl {
    Natural(NaturalController),
    Pure(PureController),
    Gru(PureGruController),
}

impl Ctl {
    pub(crate) fn load(a: &Args) -> Result<Self, String> {
        if graph_input_arity(&a.model)? == 2 {
            let policy = RecurrentOnnxPolicy::load(&a.model, 76, 128, 36)?;
            let velocity = if a.gru_host_velocity {
                if a.estimator.is_some() {
                    return Err("--gru-host-velocity と --estimator は同時に使えません".into());
                }
                eprintln!(
                    "policy: --gru-host-velocity — 速度入力に脚オドメトリを使います。\
                     GRU 契約は凍結推定器で学習しているので、これは診断用の別入力です。"
                );
                VelocitySource::Host
            } else {
                let path = a.estimator.as_deref().ok_or(
                    "GRU 契約（2 入力グラフ）には --estimator PATH が要ります\
                     （6×73 → 3 の凍結速度推定器。脚オドメトリで代用するなら \
                     --gru-host-velocity を明示してください）",
                )?;
                VelocitySource::Estimator(HistoryVelocityEstimator::load(path)?)
            };
            return PureGruController::new(policy, velocity).map(Ctl::Gru);
        }
        let mut errs = Vec::new();
        for n in [39usize, 73, 76] {
            match OnnxPolicy::load(&a.model, n) {
                Ok(p) => {
                    return if n == 39 {
                        NaturalController::new(p, a.cfg).map(Ctl::Natural)
                    } else {
                        PureController::new(p).map(Ctl::Pure)
                    };
                }
                Err(e) => errs.push(format!("{n}: {e}")),
            }
        }
        Err(format!(
            "--model は 39/73/76 入力のいずれとしても読めません — {}",
            errs.join(" / ")
        ))
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Ctl::Natural(_) => "Natural (39 入力)",
            Ctl::Pure(c) if c.wants_velocity() => "Pure (76 入力, 体速度推定つき)",
            Ctl::Pure(_) => "Pure (73 入力)",
            Ctl::Gru(c) if c.wants_host_velocity() => "GRU (76 入力 + 隠れ状態 128, 脚オドメトリ)",
            Ctl::Gru(_) => "GRU (76 入力 + 隠れ状態 128, 凍結推定器)",
        }
    }

    /// 契約ごとの学習指令域クランプ。
    pub(crate) fn clamp_fn(&self) -> fn([f64; 3]) -> [f64; 3] {
        match self {
            Ctl::Natural(_) => clamp_cmd,
            Ctl::Pure(_) => clamp_pure_cmd,
            Ctl::Gru(_) => clamp_gru_cmd,
        }
    }

    /// 契約が学習した |vx| 上限（表示と既定の安全弁の根拠に使う）。
    pub(crate) fn trained_vx_max(&self) -> f64 {
        self.clamp_fn()([1e3, 0.0, 0.0])[0]
    }

    /// 呼び出し側が体速度推定（脚オドメトリ）を渡す必要があるか。
    /// GRU で凍結推定器を使う構成では false — 速度は契約の内側で作られる。
    pub(crate) fn wants_velocity(&self) -> bool {
        match self {
            Ctl::Pure(c) => c.wants_velocity(),
            Ctl::Gru(c) => c.wants_host_velocity(),
            Ctl::Natural(_) => false,
        }
    }

    /// ネットワーク入力ではなく「しゃがみ姿勢（0.30 m）で学習した契約か」。
    /// sim の初期高さの既定に使う。
    pub(crate) fn is_crouch_contract(&self) -> bool {
        matches!(self, Ctl::Pure(_) | Ctl::Gru(_))
    }
}

/// 契約クランプ + `--vx-max` の安全弁。キーボードと初期指令の両方に通す。
pub(crate) type CmdClamp = Arc<dyn Fn([f64; 3]) -> [f64; 3] + Send + Sync>;

pub(crate) fn cmd_clamp(ctl: &Ctl, vx_max: Option<f64>) -> CmdClamp {
    let base = ctl.clamp_fn();
    match vx_max {
        Some(m) => {
            let m = m.abs();
            Arc::new(move |c| {
                let mut c = base(c);
                c[0] = c[0].clamp(-m, m);
                c
            })
        }
        None => Arc::new(move |c| base(c)),
    }
}

impl Ctl {
    pub(crate) fn default_pose_isaac(&self) -> [f64; 12] {
        match self {
            Ctl::Natural(c) => c.default_pose_isaac(),
            Ctl::Pure(c) => c.default_pose_isaac(),
            Ctl::Gru(c) => c.default_pose_isaac(),
        }
    }

    pub(crate) fn initial_gains(&self) -> (f64, f64) {
        match self {
            Ctl::Natural(c) => c.initial_gains(),
            Ctl::Pure(c) => c.initial_gains(),
            Ctl::Gru(c) => c.initial_gains(),
        }
    }

    pub(crate) fn reset(&mut self) {
        match self {
            Ctl::Natural(c) => c.reset(),
            Ctl::Pure(c) => c.reset(),
            Ctl::Gru(c) => c.reset(),
        }
    }

    pub(crate) fn gait_time_s(&self) -> f64 {
        match self {
            Ctl::Natural(c) => c.gait_time_s(),
            Ctl::Pure(c) => c.gait_time_s(),
            Ctl::Gru(c) => c.gait_time_s(),
        }
    }

    pub(crate) fn hold(&self) -> PolicyTick {
        match self {
            Ctl::Natural(c) => c.hold(),
            Ctl::Pure(c) => c.hold(),
            Ctl::Gru(c) => c.hold(),
        }
    }

    /// Natural の計画 swing。Pure / GRU は計画を持たない（呼び出し側は接地
    /// センサなり全接地なりのフォールバックを使う）。
    pub(crate) fn swing(&self) -> Option<[bool; 4]> {
        match self {
            Ctl::Natural(c) => Some(c.swing()),
            Ctl::Pure(_) | Ctl::Gru(_) => None,
        }
    }

    /// `vel_body` は体座標系の並進速度推定（Pure76 と、脚オドメトリ構成の
    /// GRU だけが消費する）。
    pub(crate) fn tick(
        &mut self,
        inp: &ObsInput,
        cmd: [f64; 3],
        vel_body: [f64; 3],
    ) -> Result<PolicyTick, String> {
        match self {
            Ctl::Natural(c) => c.tick(inp, cmd),
            Ctl::Pure(c) => c.tick(inp, cmd, vel_body),
            Ctl::Gru(c) => c.tick(inp, cmd, vel_body),
        }
    }

    /// 支持脚レンチ τ_ff。Pure / GRU 契約はフィードフォワード無し（ゼロ）。
    pub(crate) fn support_torque(
        &self,
        base: &BaseState,
        cmd: [f64; 3],
        q_isaac: &[f64; 12],
    ) -> [f64; 12] {
        match self {
            Ctl::Natural(c) => c.support_torque(base, cmd, q_isaac),
            Ctl::Pure(_) | Ctl::Gru(_) => [0.0; 12],
        }
    }
}

/// ワールド系ベクトルを体座標系へ（v_b = R(q)ᵀ v_w、q は w,x,y,z）。
pub(crate) fn world_to_body(q: &[f64; 4], v: [f64; 3]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let qv = [x, y, z];
    let dot = qv[0] * v[0] + qv[1] * v[1] + qv[2] * v[2];
    let cross = [
        qv[1] * v[2] - qv[2] * v[1],
        qv[2] * v[0] - qv[0] * v[2],
        qv[0] * v[1] - qv[1] * v[0],
    ];
    core::array::from_fn(|i| v[i] * (2.0 * w * w - 1.0) - 2.0 * w * cross[i] + 2.0 * qv[i] * dot)
}

/// テレオペの刻み。0.02 は 1 m/s 級の方策には細かすぎる（50 回押す羽目に
/// なる）ので、通常の刻みを上げたうえで Shift の粗刻みと数字キーの直接指定を
/// 用意する。上限そのものは契約のクランプ（と --vx-max）が決める。
const STEP_VX: f64 = 0.05;
const STEP_VX_COARSE: f64 = 0.20;
const STEP_VY: f64 = 0.05;
const STEP_WZ: f64 = 0.10;
/// 数字キー 1..5 に割り当てる vx。
const VX_PRESETS: [f64; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];

/// WASD テレオペ。指令は共有 cmd を書き換え、q/Esc/Ctrl-C で quit。
pub(crate) fn spawn_keyboard(
    cmd: Arc<Mutex<[f64; 3]>>,
    quit: Arc<AtomicBool>,
    clamp: CmdClamp,
) -> Result<std::thread::JoinHandle<()>, String> {
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
    enable_raw_mode().map_err(|e| format!("raw mode: {e}"))?;
    // 実効上限を出す。0.02 刻みだけだと 1.0 m/s まで 50 回押すことになり、
    // 「0.6 までしか出せない」と誤解される（実際には上限ではなく刻みの問題）。
    let vx_hi = clamp([100.0, 0.0, 0.0])[0];
    let vy_hi = clamp([0.0, 100.0, 0.0])[1];
    let wz_hi = clamp([0.0, 0.0, 100.0])[2];
    eprintln!(
        "policy: keys — W/S = vx±{STEP_VX:.2}（Shift で ±{STEP_VX_COARSE:.1}）, \
         A/D = wz±{STEP_WZ:.2}, R/F = vy±{STEP_VY:.2}\r\n\
         \x20      1..5 = vx を {P1:.1}/{P2:.1}/{P3:.1}/{P4:.1}/{P5:.1} に即設定, \
         0 か Space = 全部 0, Esc / q = 終了（伏せて抜ける）\r\n\
         \x20      指令域: vx ≤ {vx_hi:.2}, |vy| ≤ {vy_hi:.2}, |wz| ≤ {wz_hi:.2}\r",
        P1 = VX_PRESETS[0], P2 = VX_PRESETS[1], P3 = VX_PRESETS[2],
        P4 = VX_PRESETS[3], P5 = VX_PRESETS[4],
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
                        // Shift 併用で粗い刻み（1.0 m/s まで 5 回で届く）
                        KeyCode::Char('W') => c[0] += STEP_VX_COARSE,
                        KeyCode::Char('S') => c[0] -= STEP_VX_COARSE,
                        KeyCode::Up if k.modifiers.contains(KeyModifiers::SHIFT) => {
                            c[0] += STEP_VX_COARSE
                        }
                        KeyCode::Down if k.modifiers.contains(KeyModifiers::SHIFT) => {
                            c[0] -= STEP_VX_COARSE
                        }
                        // 数字キーで vx を直接指定（0 は停止）
                        KeyCode::Char(d @ '1'..='5') => {
                            c[0] = VX_PRESETS[(d as u8 - b'1') as usize]
                        }
                        KeyCode::Char('0') => *c = [0.0; 3],
                        KeyCode::Char('w') | KeyCode::Up => c[0] += STEP_VX,
                        KeyCode::Char('s') | KeyCode::Down => c[0] -= STEP_VX,
                        KeyCode::Char('a') | KeyCode::Left => c[2] += STEP_WZ,
                        KeyCode::Char('d') | KeyCode::Right => c[2] -= STEP_WZ,
                        KeyCode::Char('r') => c[1] += STEP_VY,
                        KeyCode::Char('f') => c[1] -= STEP_VY,
                        KeyCode::Char(' ') => *c = [0.0; 3],
                        KeyCode::Char('q') | KeyCode::Esc => quit.store(true, Ordering::Relaxed),
                        KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                            quit.store(true, Ordering::Relaxed)
                        }
                        _ => {}
                    }
                    *c = clamp(*c);
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
    if a.sim {
        #[cfg(feature = "sim")]
        return crate::policy_sim::run(&a);
        #[cfg(not(feature = "sim"))]
        return Err(
            "このビルドには sim が入っていません（--features sim で有効化。\
                    MUJOCO_DYNAMIC_LINK_DIR も要る）"
                .into(),
        );
    }

    let mut ctl = Ctl::load(&a)?;
    if a.odom_calibrated && !ctl.wants_velocity() {
        return Err(
            "--odom-calibrated は脚オドメトリを入力に使う契約専用です\
             （76 入力 Pure、または --gru-host-velocity の GRU）"
                .into(),
        );
    }
    if a.estimator.is_some() && !matches!(ctl, Ctl::Gru(_)) {
        return Err("--estimator は GRU 契約（2 入力グラフ）専用です".into());
    }
    eprintln!(
        "policy: {} を読み込みました — 契約: {}（Natural 時 stride {:.2}/{:.2}, height {:.2} m）",
        a.model,
        ctl.name(),
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
    let clamp = cmd_clamp(&ctl, a.vx_max);
    if let Some(m) = a.vx_max {
        eprintln!("policy: --vx-max {m:.2} — |vx| をこの値で抑えます\r");
    }
    let cmd = Arc::new(Mutex::new(clamp(a.cmd0)));
    let quit = Arc::new(AtomicBool::new(false));
    let kb = if a.keyboard {
        Some(spawn_keyboard(cmd.clone(), quit.clone(), clamp.clone())?)
    } else {
        eprintln!(
            "policy: キーボード無効。vx={:.2} vy={:.2} wz={:.2} を保持",
            a.cmd0[0], a.cmd0[1], a.cmd0[2]
        );
        None
    };

    ctl.reset();
    let mut odom = LegOdometry::new_with_planar_slip_calibration(a.odom_calibrated);
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

        // 脚オドメトリの接地マスク。Natural は計画 swing の否定でよいが、
        // Pure 契約は歩容計画を持たないので足裏力センサで測る（脚順 FL,FR,
        // RL,RR ← foot_force の FR,FL,RR,RL）。センサが無い個体では全接地に
        // 落ちる（推定速度が鈍るだけで、方策は落ちない — sim で確認済み）。
        let stance = match ctl.swing() {
            Some(sw) => [!sw[0], !sw[1], !sw[2], !sw[3]],
            None => {
                let s = plant.last_state().ok_or("LowState がありません")?;
                core::array::from_fn(|leg| {
                    s.foot_force[MISA_LEG_TO_GO2_FOOT[leg]] as f64 > CONTACT_THRESHOLD
                })
            }
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
            let vel_body = world_to_body(&inp.quat_wxyz, odom.vel_world());
            match ctl.tick(&inp, cmd_now, vel_body) {
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

#[cfg(test)]
mod stance_tests {
    use super::*;

    fn parse_ok(args: &[&str]) -> Args {
        let v: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        parse(&v).expect("parse")
    }

    /// Without a height flag the deploy standard is unchanged: 0.30 m, yaw 2.0.
    #[test]
    fn default_is_the_h30_standard() {
        let a = parse_ok(&["--model", "p.onnx"]);
        assert_eq!(a.cfg.body_height, Some(0.30));
        assert_eq!(a.cfg.yaw_stride_gain, Some(2.0));
    }

    /// --body-height recalibrates the yaw gain; the policy over-tracks yaw
    /// when crouched if it does not (133% at 0.21 m).
    #[test]
    fn body_height_recalibrates_the_yaw_gain() {
        let a = parse_ok(&["--model", "p.onnx", "--body-height", "0.24"]);
        assert_eq!(a.cfg.body_height, Some(0.24));
        assert!((a.cfg.yaw_stride_gain.unwrap() - 1.733).abs() < 1e-3);
        assert_eq!(a.cfg.stride_gain, 1.55);
    }

    /// --low-stance is the recommended 0.22 m configuration.
    #[test]
    fn low_stance_is_022() {
        let a = parse_ok(&["--model", "p.onnx", "--low-stance"]);
        assert_eq!(a.cfg.body_height, Some(0.22));
        assert!((a.cfg.yaw_stride_gain.unwrap() - 1.644).abs() < 1e-3);
    }

    /// An explicit --yaw-stride-gain wins over the schedule, in EITHER order.
    #[test]
    fn explicit_yaw_gain_wins_whatever_the_order() {
        for args in [
            vec!["--model", "p.onnx", "--body-height", "0.24", "--yaw-stride-gain", "2.0"],
            vec!["--model", "p.onnx", "--yaw-stride-gain", "2.0", "--body-height", "0.24"],
        ] {
            let a = parse_ok(&args);
            assert_eq!(a.cfg.body_height, Some(0.24), "{args:?}");
            assert_eq!(a.cfg.yaw_stride_gain, Some(2.0), "{args:?}");
        }
    }

    /// Heights outside the measured range are clamped, never extrapolated:
    /// below 0.21 m the speed envelope breaks down.
    #[test]
    fn out_of_range_height_is_clamped() {
        assert_eq!(
            parse_ok(&["--model", "p.onnx", "--body-height", "0.15"]).cfg.body_height,
            Some(0.21)
        );
        assert_eq!(
            parse_ok(&["--model", "p.onnx", "--body-height", "0.40"]).cfg.body_height,
            Some(0.31)
        );
    }
}
