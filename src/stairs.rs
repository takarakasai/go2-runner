//! MuJoCo の場に置く階段（静的な箱の列）。
//!
//! **なぜこの crate にあるか**: misa-plant-mujoco は `.misa` 一つから
//! シーン全体を組み立てるので、機体でないものは
//! [`SimOptions::extra_worldbody`] からしか入れられない。そこへ流す MJCF を
//! 作るのがここ。go2_rl 側の `make_go2_stairs.py` と**同じ形**の階段を出す
//! （登り、各段は上面が水平な箱、幅は y 方向、最上段の先は踏み場なし）。
//!
//! 寸法の既定は go2_rl の目標条件に合わせてある:
//! 蹴上げ 0.20 m × 10 段、踏面 0.30 m、幅 3.0 m。
//!
//! 階段を置いたら**接地摩擦も合わせること**。go2_rl の MuJoCo ハーネスは
//! `--foot-friction 0.8` で測っており、既定のままだと段鼻で滑って比較に
//! ならない。
//!
//! # 閉ループの食い違いは接地判定だった — 直した（2026-09-27）
//!
//! 同じ ONNX（go2_rl の Q3/model_19847）、20 cm × 10 段・幅 3 m、
//! **実物の脚オドメトリのまま**:
//!
//! | 接地の判定 | 接地本数 | 脚オドメトリの偏り | 結果 |
//! |---|---|---|---|
//! | 接触力の**世界系 z 成分**（従来） | 0.6〜2.0 本 | **−0.72 m/s** | 5 段で崩れ横へ 8 m 流れる |
//! | 接触力の**大きさ**（現在） | 1.96 本 | **+0.01 m/s** | **開始位置 4 通りとも 10 段完登** |
//!
//! **原因**: プラントは articara の `contact_force_per_foot` を使っており、
//! あれは `out[slot] += c.force_world[2]` と**世界系 z 成分だけ**を足す。
//! 法線が +z の平地では正しいが、**階段は蹴上げや段鼻で法線が傾き、
//! 符号違いの成分が相殺して接地を取りこぼす**。接地本数が落ちると
//! 脚オドメトリ（接地足の平均）が荒れ、観測 48 次元のうち胴体速度 3 次元が
//! 壊れて方策が分布外へ出る。
//!
//! 学習側（mit_rl `contact_leg_odometry_lin_vel_b`）も Python 参照ハーネスも
//! **力の大きさ**で判定している。
//!
//! **直した場所**: articara に `contact_force_magnitude_per_foot` を足し
//! （`contact_force_per_foot` は荷重用として意味を変えずに残した）、
//! misa-plant-mujoco の接地フラグをそちらに切り替えた。
//! go2-runner 側にあった暫定の上書きは外した（プラント側の修正だけで
//! **数値が完全に一致**することを確認済み: 接地 1.96 本、推定 +0.400 対
//! 真値 +0.389、開始位置 4 通りとも 10 段完登）。
//!
//! **潰した仮説（すべて外れ）**
//! - **衝突形状のメッシュ**: go2.misa はメッシュ 21 個を衝突に使うが、同じ
//!   リンクに go2.xml と同寸法のプリミティブも持ち外側を覆っている。
//!   外した `go2_primcol.misa` と比べて**平地も階段も数値が完全一致**。
//! - **トルク上限**: `--effort-limit 23.5` で学習側に揃えると**悪化**。
//! - **制御周期**: 方策 50 Hz・物理 500 Hz で Python と一致。
//! - **推定器の 10 Hz ローパス**: 切っても直らない（むしろ荒れる）。
//! - **解析 FK**: プラントの実物と **2 mm 以内**で一致（`GO2_TRACE_ODOM=1` で照合）。
//! - **ヤコビアン**: 恒等式 `ω×p + Jq̇ = R^T(v_foot − v_base)` が 0.003 m/s で成立。
//!
//! 診断は `GO2_TRACE_ODOM=1` で出る（FK 照合と恒等式の突き合わせ）。

/// `--stairs` の指定。`蹴上げ,段数,踏面,幅,開始x,摩擦,天端の踏み場` の順、後ろは省略可。
#[derive(Debug, Clone, PartialEq)]
pub struct StairsSpec {
    /// 1 段の高さ [m]。
    pub rise_m: f64,
    /// 段数。
    pub steps: usize,
    /// 踏面の奥行き [m]（進行方向）。
    pub run_m: f64,
    /// 階段の幅 [m]（横方向）。
    pub width_m: f64,
    /// 1 段目の立ち上がりの x [m]。手前に助走距離を取る。
    pub start_x_m: f64,
    /// 段の接触摩擦の滑り成分。go2_rl のハーネスの既定と同じ 0.8。
    pub friction: f64,
    /// 最上段の先に足す踏み場の奥行き [m]。0 なら足さない。
    ///
    /// **なぜ要るか**: 天端に踏み場が無いと、登りきった機体がそのまま踏み外す。
    /// 「登れたか」と「天端で踏み外したか」が混ざって判定できない
    /// （go2_rl 側も同じ理由で `make_go2_stairs.py --top-platform` を持つ）。
    pub top_platform_m: f64,
}

impl Default for StairsSpec {
    fn default() -> Self {
        Self {
            rise_m: 0.20,
            steps: 10,
            run_m: 0.30,
            width_m: 3.0,
            start_x_m: 1.2,
            friction: 0.8,
            top_platform_m: 2.0,
        }
    }
}

impl StairsSpec {
    /// `"0.20,10,0.30,3.0,1.2"` を読む。後ろの要素は省略すると既定値。
    /// 空文字は既定そのもの（`--stairs ""` ではなく `--stairs default` を想定）。
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut out = Self::default();
        if s == "default" {
            return Ok(out);
        }
        let f: Vec<&str> = s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()).collect();
        if f.is_empty() {
            return Err("--stairs の書式は 蹴上げ,段数,踏面,幅,開始x（後ろは省略可）".into());
        }
        let num = |t: &str, name: &str| -> Result<f64, String> {
            t.parse::<f64>().map_err(|_| format!("--stairs の {name} が数でありません: {t:?}"))
        };
        out.rise_m = num(f[0], "蹴上げ")?;
        if let Some(t) = f.get(1) {
            out.steps = t.parse::<usize>().map_err(|_| format!("--stairs の段数が整数でありません: {t:?}"))?;
        }
        if let Some(t) = f.get(2) { out.run_m = num(t, "踏面")?; }
        if let Some(t) = f.get(3) { out.width_m = num(t, "幅")?; }
        if let Some(t) = f.get(4) { out.start_x_m = num(t, "開始x")?; }
        if let Some(t) = f.get(5) { out.friction = num(t, "摩擦")?; }
        if let Some(t) = f.get(6) { out.top_platform_m = num(t, "天端の踏み場")?; }
        if out.rise_m <= 0.0 { return Err("--stairs の蹴上げは正の値".into()); }
        if out.steps == 0 { return Err("--stairs の段数は 1 以上".into()); }
        if out.run_m <= 0.0 { return Err("--stairs の踏面は正の値".into()); }
        if out.width_m <= 0.0 { return Err("--stairs の幅は正の値".into()); }
        Ok(self_checked(out))
    }

    /// `</worldbody>` の直前に差し込む MJCF。
    ///
    /// 各段は**地面まで届く箱**にする（薄板を浮かせると、踏み外した足が
    /// 下をすり抜けて段の内側に入り込み、接触が病的になる）。
    /// 箱は中心と半長で指定するので、n 段目（1 始まり）は
    /// 上面 z = n·rise、下面 z = 0 → 中心 z = n·rise/2、半長 z = n·rise/2。
    pub fn to_mjcf(&self) -> String {
        let hy = self.width_m * 0.5;
        let mut s = String::from("\n    <!-- go2-runner --stairs -->\n");
        for n in 1..=self.steps {
            let top = n as f64 * self.rise_m;
            let x0 = self.start_x_m + (n - 1) as f64 * self.run_m;
            // 最上段だけは踏み場を長く取らず、他と同じ踏面にする
            // （go2_rl の tgt_stairs と同じ。天端に踏み場があると
            //  「登り切った」の判定が甘くなる）。
            let hx = self.run_m * 0.5;
            let cx = x0 + hx;
            // **摩擦は明示し `priority` を付ける。** 書かないと MuJoCo は
            // 足 geom と段 geom の摩擦を混ぜるので、go2_rl の
            // `make_go2_stairs.py` が出す階段（`friction="0.8 0.02 0.01"
            // condim="6" priority="1"`）と接触が一致せず、同じ方策・同じ
            // 寸法でも結果が食い違う。**比較できることが目的**なので合わせる。
            s.push_str(&format!(
                "    <geom name=\"stair_{n}\" type=\"box\" pos=\"{cx:.4} 0 {:.4}\" \
                 size=\"{hx:.4} {hy:.4} {:.4}\" rgba=\"0.45 0.45 0.5 1\" \
                 friction=\"{mu:.3} 0.02 0.01\" condim=\"6\" priority=\"1\"/>\n",
                top * 0.5,
                top * 0.5,
                mu = self.friction
            ));
        }
        if self.top_platform_m > 0.0 {
            let top = self.steps as f64 * self.rise_m;
            let x0 = self.start_x_m + self.steps as f64 * self.run_m;
            let hx = self.top_platform_m * 0.5;
            s.push_str(&format!(
                "    <geom name=\"stair_top\" type=\"box\" pos=\"{:.4} 0 {:.4}\" \
                 size=\"{hx:.4} {hy:.4} {:.4}\" rgba=\"0.45 0.45 0.5 1\" \
                 friction=\"{mu:.3} 0.02 0.01\" condim=\"6\" priority=\"1\"/>\n",
                x0 + hx,
                top * 0.5,
                top * 0.5,
                mu = self.friction
            ));
        }
        s
    }
}

fn self_checked(s: StairsSpec) -> StairsSpec {
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 天端の踏み場は最上段と同じ高さで、階段の先に付く。
    #[test]
    fn top_platform_sits_level_with_the_last_step() {
        let s = StairsSpec { rise_m: 0.2, steps: 2, run_m: 0.3, width_m: 1.0,
                             start_x_m: 0.0, friction: 0.8, top_platform_m: 1.0 };
        let x = s.to_mjcf();
        // 最上段の上面 0.4 -> 中心 z 0.2、半長 z 0.2。x は 0.6 から 1.0 m ぶん。
        assert!(x.contains("stair_top"), "{x}");
        assert!(x.contains("pos=\"1.1000 0 0.2000\""), "{x}");
        assert!(x.contains("size=\"0.5000 0.5000 0.2000\""), "{x}");
    }

    #[test]
    fn parse_defaults_and_overrides() {
        assert_eq!(StairsSpec::parse("default").unwrap(), StairsSpec::default());
        let s = StairsSpec::parse("0.15,5").unwrap();
        assert_eq!(s.rise_m, 0.15);
        assert_eq!(s.steps, 5);
        // 省略した分は既定のまま
        assert_eq!(s.run_m, StairsSpec::default().run_m);
        let s = StairsSpec::parse("0.2,10,0.3,3.0,1.5").unwrap();
        assert_eq!(s.start_x_m, 1.5);
    }

    #[test]
    fn parse_rejects_nonsense() {
        assert!(StairsSpec::parse("").is_err());
        assert!(StairsSpec::parse("abc").is_err());
        assert!(StairsSpec::parse("0.0,3").is_err());
        assert!(StairsSpec::parse("0.2,0").is_err());
    }

    /// **各段の上面が n·rise で、箱が地面まで届いている**ことを確かめる。
    /// ここを間違えると段が浮き、踏み外した足が下へ潜る。
    #[test]
    fn boxes_reach_the_ground_and_top_out_at_n_rise() {
        let s = StairsSpec { friction: 0.8, top_platform_m: 0.0, rise_m: 0.2, steps: 3, run_m: 0.3, width_m: 3.0, start_x_m: 1.0 };
        let x = s.to_mjcf();
        // 3 段目: 上面 0.6 → 中心 z 0.3、半長 z 0.3
        assert!(x.contains("stair_3"), "{x}");
        assert!(x.contains("0.3000\" size=\"0.1500 1.5000 0.3000\""), "{x}");
        // 1 段目の x: start 1.0 + 半長 0.15 = 1.15
        assert!(x.contains("pos=\"1.1500 0 0.1000\""), "{x}");
    }

    /// 段は x 方向に踏面ぶんずつ前へ出る。
    #[test]
    fn steps_advance_by_one_run_each() {
        let s = StairsSpec { friction: 0.8, top_platform_m: 0.0, rise_m: 0.2, steps: 2, run_m: 0.3, width_m: 1.0, start_x_m: 0.0 };
        let x = s.to_mjcf();
        assert!(x.contains("pos=\"0.1500 0 0.1000\""), "{x}");
        assert!(x.contains("pos=\"0.4500 0 0.2000\""), "{x}");
    }
}
