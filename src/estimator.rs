//! 支持脚レンチ・フィードフォワードに要る胴体状態（高さ・世界系速度）の
//! 脚オドメトリ推定。
//!
//! 接地している足は世界に対して止まっている、という仮定から:
//!
//! ```text
//! 0 = v_base_w + R (ω × p_foot_b + J q̇)
//! ⇒ v_base_w = −R (ω × p_foot_b + J q̇)       … 接地足で平均
//! 高さ    = −mean_z( R p_foot_b )              … 接地足で平均
//! ```
//!
//! misa-runner の `BodyEstimator` と同じ考え方の最小構成（あちらは `.misa`
//! モデルの FK を使う。ここは misa-policy-runner の解析 FK — policy モードは
//! モデルファイルに依存しない）。速度は一次 LPF（既定 10 Hz）で均す。
//!
//! 接地の判定は**方策側の計画**（軌道の swing の否定）を使う。歩き出しの
//! 計画一致は良く、足裏力は離地直後に遅れて残るため。

use misa_policy_runner::go2::GO2_TO_ISAAC;
use misa_policy_runner::natural::{fk, leg_jacobian};
use misa_policy_runner::support::quat_rotate_inverse;

/// `R(q) v` — 胴体系ベクトルを世界系へ。
fn quat_rotate(q: &[f64; 4], v: [f64; 3]) -> [f64; 3] {
    // R v = (R^T)^{-1} v。共役クォータニオンで inverse 回転を使う。
    let conj = [q[0], -q[1], -q[2], -q[3]];
    quat_rotate_inverse(&conj, v)
}

pub struct LegOdometry {
    lpf_alpha_hz: f64,
    vel_world: [f64; 3],
    vel_valid: bool,
    height_m: f64,
    /// Latest per-leg base-velocity candidates before stance aggregation.
    candidates_world: [Option<[f64; 3]>; 4],
    /// Empirical forward-slip calibration, enabled explicitly by the CLI.
    planar_slip_calibration: bool,
}

impl LegOdometry {
    pub fn new() -> Self {
        Self {
            lpf_alpha_hz: 10.0,
            vel_world: [0.0; 3],
            vel_valid: false,
            height_m: 0.30,
            candidates_world: [None; 4],
            planar_slip_calibration: false,
        }
    }

    pub fn new_with_planar_slip_calibration(enabled: bool) -> Self {
        let mut estimator = Self::new();
        estimator.planar_slip_calibration = enabled;
        estimator
    }

    /// 1 周期ぶん更新する。`q_go2`/`dq_go2` は Go2 モータ順、`stance` は
    /// FL,FR,RL,RR。接地足が無い周期は前回値を保持する。
    pub fn update(
        &mut self,
        quat_wxyz: &[f64; 4],
        gyro: [f64; 3],
        q_go2: &[f64; 12],
        dq_go2: &[f64; 12],
        stance: [bool; 4],
        dt: f64,
    ) {
        self.candidates_world = [None; 4];
        let mut q_isaac = [0.0f64; 12];
        let mut dq_isaac = [0.0f64; 12];
        for g in 0..12 {
            q_isaac[GO2_TO_ISAAC[g]] = q_go2[g];
            dq_isaac[GO2_TO_ISAAC[g]] = dq_go2[g];
        }
        let feet = fk(&q_isaac);
        let mut v_sum = [0.0f64; 3];
        let mut h_sum = 0.0f64;
        let mut n = 0usize;
        for l in 0..4 {
            if !stance[l] {
                continue;
            }
            let p = feet[l];
            let j = leg_jacobian(&q_isaac, l);
            let dq_leg = [dq_isaac[l], dq_isaac[4 + l], dq_isaac[8 + l]];
            // 胴体系の足速度: ω × p + J q̇
            let jqd = [
                j[0][0] * dq_leg[0] + j[0][1] * dq_leg[1] + j[0][2] * dq_leg[2],
                j[1][0] * dq_leg[0] + j[1][1] * dq_leg[1] + j[1][2] * dq_leg[2],
                j[2][0] * dq_leg[0] + j[2][1] * dq_leg[1] + j[2][2] * dq_leg[2],
            ];
            let wxp = [
                gyro[1] * p[2] - gyro[2] * p[1],
                gyro[2] * p[0] - gyro[0] * p[2],
                gyro[0] * p[1] - gyro[1] * p[0],
            ];
            let foot_vel_b = [jqd[0] + wxp[0], jqd[1] + wxp[1], jqd[2] + wxp[2]];
            let v_w = quat_rotate(quat_wxyz, foot_vel_b);
            self.candidates_world[l] = Some([-v_w[0], -v_w[1], -v_w[2]]);
            for k in 0..3 {
                v_sum[k] -= v_w[k];
            }
            h_sum -= quat_rotate(quat_wxyz, p)[2];
            n += 1;
        }
        if n == 0 {
            return; // 全脚遊脚: 前回値を保持
        }
        let v_now = [
            v_sum[0] / n as f64,
            v_sum[1] / n as f64,
            v_sum[2] / n as f64,
        ];
        let h_now = h_sum / n as f64;
        if !self.vel_valid {
            self.vel_world = v_now;
            self.height_m = h_now;
            self.vel_valid = true;
            return;
        }
        let alpha = if dt > 0.0 {
            let tau = 1.0 / (2.0 * std::f64::consts::PI * self.lpf_alpha_hz);
            (dt / (tau + dt)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        for k in 0..3 {
            self.vel_world[k] += alpha * (v_now[k] - self.vel_world[k]);
        }
        self.height_m += alpha * (h_now - self.height_m);
    }

    pub fn vel_world(&self) -> [f64; 3] {
        if !self.planar_slip_calibration {
            return self.vel_world;
        }
        let mut corrected = self.vel_world;
        let speed_x = corrected[0].abs();
        if speed_x > 1e-6 {
            // Fit through the origin from ten MuJoCo steady forward-speed
            // points (0.05--0.50 m/s command). Lateral velocity is left
            // untouched until equivalent vy data exists. Above the measured
            // raw-odom range, hold the boundary gain instead of extrapolating.
            let r = speed_x.min(0.377);
            let gain = 2.22419053 - 4.95014345 * r + 6.74763747 * r * r;
            corrected[0] *= gain;
        }
        corrected
    }

    pub fn height_m(&self) -> f64 {
        self.height_m
    }

    pub fn candidates_world(&self) -> [Option<[f64; 3]>; 4] {
        self.candidates_world
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use misa_policy_runner::go2::ISAAC_TO_GO2;
    use misa_policy_runner::natural::{ik, trajectory, TrajectoryCfg};

    /// 静止立位（q̇ = 0、ω = 0、水平）: 速度 0、高さ = 立ち高さ − 足球
    /// 中心のオフセット。
    #[test]
    fn standing_still_reads_zero_velocity_and_the_stance_height() {
        let cfg = TrajectoryCfg::h30_standard();
        let r = trajectory(0.0, [0.0; 3], &cfg);
        let q_isaac = ik(&r.feet);
        let mut q_go2 = [0.0f64; 12];
        for i in 0..12 {
            q_go2[ISAAC_TO_GO2[i]] = q_isaac[i];
        }
        let mut est = LegOdometry::new();
        est.update(
            &[1.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            &q_go2,
            &[0.0; 12],
            [true; 4],
            0.002,
        );
        let v = est.vel_world();
        assert!(v.iter().all(|x| x.abs() < 1e-12), "{v:?}");
        // trajectory の足 z は .023 − height（足球中心）なので、FK からの
        // 高さは height − 0.023。
        assert!(
            (est.height_m() - (0.30 - 0.023)).abs() < 1e-9,
            "{}",
            est.height_m()
        );
    }

    /// 胴体が +x に動くとき（足は世界に固定）、関節速度から −x 向きの
    /// 足速度が見え、推定は +x の胴体速度を返す。
    #[test]
    fn forward_motion_is_recovered_from_joint_rates() {
        let cfg = TrajectoryCfg::h30_standard();
        let r = trajectory(0.0, [0.0; 3], &cfg);
        let q_isaac = ik(&r.feet);
        // 数値微分: 足を −x に動かす IK の変化率 ≈ J⁻¹ ẋ
        let vx = 0.1;
        let dt = 1e-6;
        let mut feet2 = r.feet;
        for l in 0..4 {
            feet2[l][0] -= vx * dt;
        }
        let q2 = ik(&feet2);
        let mut q_go2 = [0.0f64; 12];
        let mut dq_go2 = [0.0f64; 12];
        for i in 0..12 {
            q_go2[ISAAC_TO_GO2[i]] = q_isaac[i];
            dq_go2[ISAAC_TO_GO2[i]] = (q2[i] - q_isaac[i]) / dt;
        }
        let mut est = LegOdometry::new();
        est.update(
            &[1.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            &q_go2,
            &dq_go2,
            [true; 4],
            0.002,
        );
        let v = est.vel_world();
        assert!((v[0] - vx).abs() < 1e-4, "vx {}", v[0]);
        assert!(v[1].abs() < 1e-6 && v[2].abs() < 1e-4, "{v:?}");
    }
}
