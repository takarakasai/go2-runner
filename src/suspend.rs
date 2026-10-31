//! 吊った状態の診断 — 立ち上げ手順の**手順 4・5**。
//!
//! 手順 4（吊って指令 0）と手順 5（吊って `--vx 0.3`）は、**接地させる前に
//! 発振していないかを見る唯一の関門**で、合否は「脚が静かか」「歩容が出て
//! 左右が揃っているか」という目で見る判断だった。目で見る判断は
//! 記録に残らず、実機と sim を突き合わせられない。ここは数で出す。
//!
//! 出すのは 4 つ:
//!   1. 既定姿勢からのずれ（平均・最大） — 保持できているか
//!   2. `|dq|` の最大と実効値 — 暴れていないか
//!   3. **振動の周波数**（`dq` の符号反転をデッドバンド付きで数える）
//!      — 脚の PD が唸っているか（唸りは数 10 Hz、歩容は 1〜3 Hz。
//!        **この 2 つは `|dq|max` では見分けられない**ので周波数で分ける）
//!   4. 左右差（前 FR/FL、後 RR/RL の可動域比と歩調差）
//!
//! 接地していないので脚の慣性は空荷で、Kp 25 / Kd 0.5 の MIT ゲインが
//! 接地時と同じ余裕を持つとは限らない。**吊りは接地より唸りやすい側**。
//!
//! 実機でも sim（`policy --sim --suspend`）でも同じ関数を通すので、
//! 数字がそのまま比べられる。

/// 符号反転を数えるときのデッドバンド [rad/s]。これ未満の `dq` は
/// 「止まっている」とみなす。入れないとセンサ雑音と数値誤差で
/// 毎周期符号が変わり、静止している関節が「250 Hz で振動」になる。
const DQ_BAND: f64 = 0.5;

/// 歩容と唸りの境目 [Hz]。学習側の歩調は 1〜3 Hz 程度で、関節の PD の
/// 唸りはそれより 1 桁上に出る。
const BUZZ_HZ: f64 = 8.0;

/// 起動の蹴りを表から外す秒数。
///
/// 方策は既定姿勢から始まるので最初の数 100 ms に大きな `dq` が 1 回出る。
/// これを定常の統計に混ぜると、**落ち着いた方策と暴れ続ける方策が同じ
/// `|dq|max` に見える**（実測: P は 8.9 rad/s の蹴り 1 回のあとほぼ静止、
/// ZC は 24 rad/s で振動し続けたが、どちらも「|dq|max が大きい」で並ぶ）。
/// 蹴りは別枠で出す — 手を離す前に知りたい量なので捨てない。
const WARMUP_S: f64 = 2.0;

/// 手順 4（指令 0）の合格線。sim 実測（ZC・P とも |dq|max 0.1 rad/s 未満、
/// 振動 0 Hz）に対して十分な余裕を取った値。
const QUIET_DQ_MAX: f64 = 2.0;
const QUIET_DEV_MAX: f64 = 0.10;

/// Go2 SDK の関節名（FR, FL, RR, RL × hip/thigh/calf）。
pub const GO2_JOINT_NAMES: [&str; 12] = [
    "FR hip", "FR thigh", "FR calf",
    "FL hip", "FL thigh", "FL calf",
    "RR hip", "RR thigh", "RR calf",
    "RL hip", "RL thigh", "RL calf",
];

/// 吊り診断の集計器。`q` / `dq` はすべて **Go2 SDK 順**。
pub struct SuspendReport {
    default_go2: [f64; 12],
    dev_sum: [f64; 12],
    dev_abs_max: [f64; 12],
    dq_abs_max: [f64; 12],
    dq_sq_sum: [f64; 12],
    q_min: [f64; 12],
    q_max: [f64; 12],
    /// デッドバンドを超えた直近の `dq` の符号（0 = まだ超えていない）。
    sign: [i8; 12],
    /// 符号反転の回数（= 半周期の数）。
    flips: [u32; 12],
    n: u64,
    t: f64,
    /// 経過時間（`WARMUP_S` の判定用。`t` は定常窓の長さなので別に持つ）。
    elapsed: f64,
    /// 起動 `WARMUP_S` 秒間の `|dq|` 最大とその関節。
    kick_dq: f64,
    kick_joint: usize,
}

impl SuspendReport {
    /// `default_go2` は方策の既定姿勢（Go2 SDK 順）。
    pub fn new(default_go2: [f64; 12]) -> Self {
        Self {
            default_go2,
            dev_sum: [0.0; 12],
            dev_abs_max: [0.0; 12],
            dq_abs_max: [0.0; 12],
            dq_sq_sum: [0.0; 12],
            q_min: [f64::INFINITY; 12],
            q_max: [f64::NEG_INFINITY; 12],
            sign: [0; 12],
            flips: [0; 12],
            n: 0,
            t: 0.0,
            elapsed: 0.0,
            kick_dq: 0.0,
            kick_joint: 0,
        }
    }

    pub fn update(&mut self, q_go2: &[f64; 12], dq_go2: &[f64; 12], dt: f64) {
        self.elapsed += dt;
        if self.elapsed < WARMUP_S {
            for j in 0..12 {
                if dq_go2[j].abs() > self.kick_dq {
                    self.kick_dq = dq_go2[j].abs();
                    self.kick_joint = j;
                }
            }
            return;
        }
        for j in 0..12 {
            let dev = q_go2[j] - self.default_go2[j];
            self.dev_sum[j] += dev;
            self.dev_abs_max[j] = self.dev_abs_max[j].max(dev.abs());
            self.dq_abs_max[j] = self.dq_abs_max[j].max(dq_go2[j].abs());
            self.dq_sq_sum[j] += dq_go2[j] * dq_go2[j];
            self.q_min[j] = self.q_min[j].min(q_go2[j]);
            self.q_max[j] = self.q_max[j].max(q_go2[j]);
            if dq_go2[j].abs() > DQ_BAND {
                let s = if dq_go2[j] > 0.0 { 1i8 } else { -1i8 };
                if self.sign[j] != 0 && s != self.sign[j] {
                    self.flips[j] += 1;
                }
                self.sign[j] = s;
            }
        }
        self.n += 1;
        self.t += dt;
    }

    /// 関節 `j` の振動周波数 [Hz]。符号反転 2 回で 1 周期。
    pub fn freq_hz(&self, j: usize) -> f64 {
        if self.t <= 0.0 {
            return 0.0;
        }
        self.flips[j] as f64 / (2.0 * self.t)
    }

    pub fn dq_rms(&self, j: usize) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        (self.dq_sq_sum[j] / self.n as f64).sqrt()
    }

    pub fn range(&self, j: usize) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.q_max[j] - self.q_min[j]
    }

    /// 全関節の `|dq|` 最大と、その関節。
    pub fn worst_dq(&self) -> (usize, f64) {
        let mut j0 = 0;
        for j in 1..12 {
            if self.dq_abs_max[j] > self.dq_abs_max[j0] {
                j0 = j;
            }
        }
        (j0, self.dq_abs_max[j0])
    }

    /// 最も速く振動している関節と、その周波数。
    pub fn worst_freq(&self) -> (usize, f64) {
        let mut j0 = 0;
        for j in 1..12 {
            if self.freq_hz(j) > self.freq_hz(j0) {
                j0 = j;
            }
        }
        (j0, self.freq_hz(j0))
    }

    /// 起動 `WARMUP_S` 秒の `|dq|` 最大とその関節名。
    pub fn kick(&self) -> (&'static str, f64) {
        (GO2_JOINT_NAMES[self.kick_joint], self.kick_dq)
    }

    /// 走行中に 0.5 s ごとに出す 1 行（live）。
    pub fn live_line(&self) -> String {
        if self.n == 0 {
            // 定常窓に入る前は蹴りを出す（空欄より役に立つ）。
            let (kj, kdq) = self.kick();
            return format!("吊り: 起動中 蹴り |dq|max {kdq:.2} rad/s（{kj}）");
        }
        let (jd, dq) = self.worst_dq();
        let (jf, f) = self.worst_freq();
        format!(
            "吊り: |dq|max {dq:.2} rad/s（{}） 振動 {f:.1} Hz（{}）",
            GO2_JOINT_NAMES[jd], GO2_JOINT_NAMES[jf]
        )
    }

    /// 手順 4 の合否（指令 0 で静かに保持できているか）。
    /// 返すのは `Err(理由)` か `Ok(())`。
    pub fn quiet_verdict(&self) -> Result<(), String> {
        let (jd, dq) = self.worst_dq();
        let (jf, f) = self.worst_freq();
        let mut bad = Vec::new();
        if self.n == 0 {
            return Err(format!(
                "定常窓がありません（{WARMUP_S:.0} s より短い走行では判定できない）"
            ));
        }
        if dq > QUIET_DQ_MAX {
            bad.push(format!(
                "{} の |dq| が {dq:.2} rad/s（合格線 {QUIET_DQ_MAX:.1}）",
                GO2_JOINT_NAMES[jd]
            ));
        }
        if f > BUZZ_HZ {
            bad.push(format!(
                "{} が {f:.1} Hz で振動（歩容の帯 {BUZZ_HZ:.0} Hz 以上は唸り）",
                GO2_JOINT_NAMES[jf]
            ));
        }
        let dev = (0..12).fold(0.0f64, |m, j| m.max(self.dev_abs_max[j]));
        if dev > QUIET_DEV_MAX {
            bad.push(format!(
                "既定姿勢から {dev:.3} rad 離れた（合格線 {QUIET_DEV_MAX:.2}）"
            ));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(bad.join(" / "))
        }
    }

    /// 脚（0=FR, 1=FL, 2=RR, 3=RL）の thigh の歩調 [Hz] と可動域 [rad]。
    /// **歩容があるかは thigh で見る** — hip はほぼ動かず、calf は
    /// thigh と同位相で動くので、1 本選ぶなら thigh。
    pub fn leg_gait(&self, leg: usize) -> (f64, f64) {
        let j = leg * 3 + 1;
        (self.freq_hz(j), self.range(j))
    }

    /// 左右差。`(前の可動域比, 前の歩調差, 後の可動域比, 後の歩調差)`。
    /// 可動域比は小さい側 / 大きい側なので 1.0 が完全対称。
    pub fn symmetry(&self) -> (f64, f64, f64, f64) {
        let ratio = |a: f64, b: f64| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if hi > 1e-9 {
                lo / hi
            } else {
                1.0
            }
        };
        let (f_fr, r_fr) = self.leg_gait(0);
        let (f_fl, r_fl) = self.leg_gait(1);
        let (f_rr, r_rr) = self.leg_gait(2);
        let (f_rl, r_rl) = self.leg_gait(3);
        (
            ratio(r_fr, r_fl),
            (f_fr - f_fl).abs(),
            ratio(r_rr, r_rl),
            (f_rr - f_rl).abs(),
        )
    }

    /// 走行後の表。`tag` は "policy" / "policy-sim"。
    pub fn print(&self, tag: &str) {
        let (kj, kdq) = self.kick();
        if self.n == 0 {
            eprintln!(
                "{tag}: 吊り診断 — 走行が {WARMUP_S:.0} s 未満なので定常窓がありません\
                 （起動の蹴り |dq|max {kdq:.2} rad/s（{kj}）のみ）"
            );
            return;
        }
        eprintln!(
            "{tag}: 吊り診断 定常 {:.1} s / {} tick（起動 {WARMUP_S:.0} s は別枠。\
             ずれ・可動域は rad）",
            self.t, self.n
        );
        eprintln!("{tag}: 起動の蹴り（t<{WARMUP_S:.0} s）|dq|max {kdq:.2} rad/s（{kj}）");
        eprintln!(
            "{tag}:   関節       ずれ平均  |ずれ|max  |dq|max  dq実効  振動[Hz]  可動域"
        );
        for j in 0..12 {
            eprintln!(
                "{tag}:   {:<9} {:+8.3}  {:9.3}  {:7.2}  {:6.2}  {:8.1}  {:6.3}",
                GO2_JOINT_NAMES[j],
                self.dev_sum[j] / self.n as f64,
                self.dev_abs_max[j],
                self.dq_abs_max[j],
                self.dq_rms(j),
                self.freq_hz(j),
                self.range(j),
            );
        }
        let legs = ["FR", "FL", "RR", "RL"];
        let gait: Vec<String> = (0..4)
            .map(|l| {
                let (f, r) = self.leg_gait(l);
                format!("{} {f:.2} Hz/{r:.3} rad", legs[l])
            })
            .collect();
        eprintln!("{tag}: 歩容（thigh の歩調/可動域）  {}", gait.join("  "));
        let (rf, df, rr, dr) = self.symmetry();
        eprintln!(
            "{tag}: 左右差  前 FR/FL 可動域比 {rf:.2}・歩調差 {df:.2} Hz / \
             後 RR/RL 可動域比 {rr:.2}・歩調差 {dr:.2} Hz"
        );
        match self.quiet_verdict() {
            Ok(()) => eprintln!(
                "{tag}: 静止判定 OK（手順 4 の合格線を満たす。手順 5 では歩容が出るので\
                 この判定は意味を持たない）"
            ),
            Err(why) => eprintln!("{tag}: 静止判定 NG — {why}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(freq: f64, amp: f64, secs: f64) -> SuspendReport {
        let mut r = SuspendReport::new([0.0; 12]);
        let dt = 0.002;
        let n = (secs / dt) as u64;
        for k in 0..n {
            let t = k as f64 * dt;
            let w = 2.0 * std::f64::consts::PI * freq;
            let q = [amp * (w * t).sin(); 12];
            let dq = [amp * w * (w * t).cos(); 12];
            r.update(&q, &dq, dt);
        }
        r
    }

    /// 既知の正弦の周波数を取り戻せる（これが合わないと唸りと歩容を
    /// 見分けられない）。
    #[test]
    fn recovers_the_frequency_of_a_sine() {
        let r = run(2.0, 0.3, 10.0);
        assert!((r.freq_hz(0) - 2.0).abs() < 0.1, "{}", r.freq_hz(0));
        let r = run(40.0, 0.05, 10.0);
        assert!((r.freq_hz(0) - 40.0).abs() < 1.0, "{}", r.freq_hz(0));
    }

    /// **静止は 0 Hz。** デッドバンドが無いと雑音で符号が毎周期変わり、
    /// 静止が「250 Hz の振動」に化ける。
    #[test]
    fn a_still_joint_with_noise_reads_zero_hz() {
        let mut r = SuspendReport::new([0.0; 12]);
        for k in 0..5000 {
            // ±0.2 rad/s の雑音（デッドバンド 0.5 未満）。
            let s = if k % 2 == 0 { 0.2 } else { -0.2 };
            r.update(&[0.0; 12], &[s; 12], 0.002);
        }
        assert_eq!(r.freq_hz(0), 0.0);
        assert!(r.quiet_verdict().is_ok());
    }

    /// 唸りは静止判定で落ちる。
    #[test]
    fn a_buzzing_joint_fails_the_quiet_verdict() {
        let r = run(40.0, 0.05, 5.0);
        let e = r.quiet_verdict().unwrap_err();
        assert!(e.contains("Hz"), "{e}");
    }

    /// **起動の蹴りは定常の表に入らない。** ここが混ざると「蹴って
    /// 静まる方策」と「暴れ続ける方策」が同じ |dq|max に見える。
    #[test]
    fn the_startup_kick_stays_out_of_the_steady_window() {
        let mut r = SuspendReport::new([0.0; 12]);
        // 0〜1 s: 20 rad/s の蹴り。1 s 以降（定常窓は 2 s 以降）: 静止。
        for k in 0..5000 {
            let t = k as f64 * 0.002;
            let dq = if t < 1.0 { [20.0; 12] } else { [0.0; 12] };
            r.update(&[0.0; 12], &dq, 0.002);
        }
        assert_eq!(r.kick().1, 20.0);
        assert_eq!(r.worst_dq().1, 0.0, "定常窓に蹴りが漏れている");
        assert!(r.quiet_verdict().is_ok());
    }

    /// 左右対称の歩容は比 1.0・歩調差 0。
    #[test]
    fn a_symmetric_gait_reads_as_symmetric() {
        let r = run(2.0, 0.3, 10.0);
        let (rf, df, rr, dr) = r.symmetry();
        assert!((rf - 1.0).abs() < 1e-9 && (rr - 1.0).abs() < 1e-9);
        assert!(df < 1e-9 && dr < 1e-9);
    }
}
