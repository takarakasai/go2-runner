//! Go2Plant — Unitree Go2 の rt/lowcmd / rt/lowstate を misa-core の
//! [`Plant`] に載せるブリッジ。
//!
//! 対応表（misa の軸順は FL,FR,RL,RR × hip/thigh/calf、Go2 SDK は
//! FR,FL,RR,RL × hip/thigh/calf）:
//!
//! | misa 脚 | Go2 モータ基点 |
//! |---------|---------------|
//! | FL (0)  | 3             |
//! | FR (1)  | 0             |
//! | RL (2)  | 9             |
//! | RR (3)  | 6             |
//!
//! 符号・ゼロ点の変換は**無い**: LowState の関節角は URDF の関節角そのもの
//! で、go2.misa も同じ URDF から生成されている。
//!
//! モードの写像:
//!   - `Idle`      → kp=0, kd=0, τ=0（脱力）
//!   - `Position`  → q=目標, kp/kd はプロファイルの `mit_gains`（関節種別）
//!   - `Impedance` → q/dq/kp/kd/τ_ff を 1:1（MIT。RL 方策はこれ）
//!   - `Torque`    → q=PosStopF, dq=VelStopF, kp=kd=0, τ
//!   - `Velocity`  → 受け付けない（capabilities に載せない）

use std::time::{Duration, Instant};

use misa_runner::misa_core::{
    Axis, AxisRole, AxisTable, Command, ControlMode, Imu, Observation, Plant, PlantCaps, Time,
};
use unitree_go2::{
    init_lowcmd, joint, set_crc, topics, LowCmd, LowState, Participant, Reader, ReaderQos,
    Writer, WriterQos, POS_STOP_F, VEL_STOP_F,
};

/// misa 軸 index（FL,FR,RL,RR × 3）→ Go2 SDK モータ index。
pub const MISA_TO_GO2: [usize; 12] = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8];

/// misa 脚 slot（FL,FR,RL,RR）→ Go2 `foot_force` の添字（FR,FL,RR,RL）。
pub const MISA_LEG_TO_GO2_FOOT: [usize; 4] = [1, 0, 3, 2];

/// 足裏力センサの接地判定しきい値（生値 ≈ N）。
pub const CONTACT_THRESHOLD: f64 = 20.0;

/// 観測がこの時間を超えて更新されない場合は `valid` を疑う目安。
/// （misa-runner 側の SafetyGate は age を見て自分で判断する。）
const STATE_WAIT: Duration = Duration::from_secs(5);

pub struct Go2Plant {
    axes: AxisTable,
    caps: PlantCaps,
    writer: Writer<LowCmd>,
    reader: Reader<LowState>,
    lowcmd: LowCmd,
    /// Position モードに使う kp/kd（`[hip, thigh, calf]`）。
    default_kp: [f64; 3],
    default_kd: [f64; 3],
    epoch: Instant,
    last_state: Option<LowState>,
    last_state_at: Instant,
    armed: bool,
    // DDS の participant は writer/reader より長生きでなければならない。
    _dp: Participant,
}

impl Go2Plant {
    /// `iface` は機体へ届く NIC 名（例 "eth0"）。`None` なら既定の経路。
    pub fn connect(
        iface: Option<&str>,
        default_kp: [f64; 3],
        default_kd: [f64; 3],
    ) -> Result<Self, String> {
        let dp = Participant::new(0, iface).map_err(|e| format!("DDS participant: {e}"))?;
        let cmd_topic = dp
            .create_topic::<LowCmd>(topics::LOW_CMD)
            .map_err(|e| format!("cmd topic: {e}"))?;
        let writer = dp
            .create_writer(&cmd_topic, WriterQos::low_level_default())
            .map_err(|e| format!("writer: {e}"))?;
        let state_topic = dp
            .create_topic::<LowState>(topics::LOW_STATE)
            .map_err(|e| format!("state topic: {e}"))?;
        let reader = dp
            .create_reader(&state_topic, ReaderQos::low_level_default())
            .map_err(|e| format!("reader: {e}"))?;

        let mut axes = Vec::with_capacity(12);
        for (leg, leg_name) in ["FL", "FR", "RL", "RR"].iter().enumerate() {
            for (j, kind) in ["hip", "thigh", "calf"].iter().enumerate() {
                axes.push(Axis {
                    name: format!("{leg_name}_{kind}_joint"),
                    role: AxisRole::Leg {
                        leg: leg as u8,
                        joint: j as u8,
                    },
                });
            }
        }
        let axes = AxisTable::new(axes)?;
        let caps = PlantCaps {
            modes: vec![ControlMode::Position, ControlMode::Impedance, ControlMode::Torque],
            has_imu: true,
            has_contacts: true,
            driven: vec![true; 12],
        };

        let mut plant = Self {
            axes,
            caps,
            writer,
            reader,
            lowcmd: init_lowcmd(),
            default_kp,
            default_kd,
            epoch: Instant::now(),
            last_state: None,
            last_state_at: Instant::now(),
            armed: false,
            _dp: dp,
        };
        // 最初の LowState を待つ。来なければ「起動条件が整っていない」。
        let deadline = Instant::now() + STATE_WAIT;
        loop {
            if plant.poll_state()? {
                break;
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "{}LowState が届きません（iface / 192.168.123.x / 配線を確認）",
                    misa_runner::runner::RETRYABLE
                ));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        Ok(plant)
    }

    /// 新しい LowState があれば取り込む。`Ok(true)` = 何か受けた。
    pub fn poll_state(&mut self) -> Result<bool, String> {
        let mut got = false;
        // 溜まっていれば最新まで読み飛ばす（観測は常に最新の 1 つ）。
        while let Some(s) = self.reader.poll().map_err(|e| format!("poll: {e}"))? {
            self.last_state = Some(s);
            self.last_state_at = Instant::now();
            got = true;
        }
        Ok(got)
    }

    /// 直近の生の LowState（policy モードが直接使う: クォータニオン、
    /// 加速度、足裏力は misa の [`Observation`] に収まらない）。
    pub fn last_state(&self) -> Option<&LowState> {
        self.last_state.as_ref()
    }

    /// 直近の LowState の古さ。
    pub fn state_age(&self) -> Duration {
        self.last_state_at.elapsed()
    }

    /// 全モータ脱力の 1 フレームを送る。
    pub fn send_limp(&mut self) -> Result<(), String> {
        for j in 0..joint::NUM_LEG_JOINTS {
            let m = &mut self.lowcmd.motor_cmd[j];
            m.q = 0.0;
            m.dq = 0.0;
            m.kp = 0.0;
            m.kd = 0.0;
            m.tau = 0.0;
        }
        set_crc(&mut self.lowcmd);
        self.writer
            .write(&self.lowcmd)
            .map_err(|e| format!("write: {e}"))
    }

    /// 生の LowCmd を送る（policy モード用の直接口）。`motor` は Go2 順。
    /// misa の Plant 契約の外なので、`run` 経路では使われない。
    pub fn send_raw(
        &mut self,
        q: &[f64; 12],
        dq: &[f64; 12],
        kp: &[f64; 12],
        kd: &[f64; 12],
        tau: &[f64; 12],
    ) -> Result<(), String> {
        for j in 0..joint::NUM_LEG_JOINTS {
            let m = &mut self.lowcmd.motor_cmd[j];
            m.q = q[j] as f32;
            m.dq = dq[j] as f32;
            m.kp = kp[j] as f32;
            m.kd = kd[j] as f32;
            m.tau = tau[j] as f32;
        }
        set_crc(&mut self.lowcmd);
        self.writer
            .write(&self.lowcmd)
            .map_err(|e| format!("write: {e}"))
    }

    fn fill_observation(&self, obs: &mut Observation) {
        let now = self.epoch.elapsed();
        obs.time = Time::from_secs_f64(now.as_secs_f64());
        let Some(s) = self.last_state.as_ref() else {
            return;
        };
        let age = self.last_state_at.elapsed();
        for i in 0..12 {
            let g = MISA_TO_GO2[i];
            if let Some(a) = obs.get_mut(misa_runner::misa_core::AxisId::new(i as u16)) {
                a.position_rad = s.motor_state[g].q as f64;
                a.velocity_rad_s = s.motor_state[g].dq as f64;
                a.torque_nm = None;
                a.health.valid = true;
                a.health.age = age;
                a.health.fault_raw = 0;
                a.health.temperature_c = None;
                a.health.voltage_v = None;
            }
        }
        let im = &s.imu_state;
        obs.imu = Some(Imu {
            rpy_rad: [im.rpy[0] as f64, im.rpy[1] as f64, im.rpy[2] as f64],
            gyro_rad_s: [
                im.gyroscope[0] as f64,
                im.gyroscope[1] as f64,
                im.gyroscope[2] as f64,
            ],
            accel_m_s2: [
                im.accelerometer[0] as f64,
                im.accelerometer[1] as f64,
                im.accelerometer[2] as f64,
            ],
            age,
        });
        for leg in 0..4 {
            let f = s.foot_force[MISA_LEG_TO_GO2_FOOT[leg]] as f64;
            obs.contacts[leg] = Some(f > CONTACT_THRESHOLD);
        }
    }
}

impl Plant for Go2Plant {
    fn axes(&self) -> &AxisTable {
        &self.axes
    }

    fn capabilities(&self) -> &PlantCaps {
        &self.caps
    }

    fn arm(&mut self) -> Result<(), String> {
        self.armed = true;
        Ok(())
    }

    fn disarm(&mut self) -> Result<(), String> {
        self.armed = false;
        self.send_limp()
    }

    fn status_line(&self) -> String {
        format!("lowstate age {:.0} ms", self.state_age().as_secs_f64() * 1e3)
    }

    fn exchange(&mut self, cmd: &Command, obs: &mut Observation) -> Result<(), String> {
        for (i, a) in cmd.axes().iter().enumerate() {
            let g = MISA_TO_GO2[i];
            let kind = i % 3; // hip / thigh / calf
            let m = &mut self.lowcmd.motor_cmd[g];
            match a.mode {
                ControlMode::Idle => {
                    m.q = 0.0;
                    m.dq = 0.0;
                    m.kp = 0.0;
                    m.kd = 0.0;
                    m.tau = 0.0;
                }
                ControlMode::Position => {
                    m.q = a.position_rad as f32;
                    m.dq = 0.0;
                    m.kp = self.default_kp[kind] as f32;
                    m.kd = self.default_kd[kind] as f32;
                    m.tau = 0.0;
                }
                ControlMode::Impedance => {
                    m.q = a.position_rad as f32;
                    m.dq = a.velocity_rad_s as f32;
                    m.kp = a.kp_nm_per_rad as f32;
                    m.kd = a.kd_nm_s_per_rad as f32;
                    m.tau = a.torque_ff_nm as f32;
                }
                ControlMode::Torque => {
                    m.q = POS_STOP_F;
                    m.dq = VEL_STOP_F;
                    m.kp = 0.0;
                    m.kd = 0.0;
                    m.tau = a.torque_ff_nm as f32;
                }
                ControlMode::Velocity => {
                    return Err("Go2Plant は Velocity 制御を受け付けません".into());
                }
            }
        }
        set_crc(&mut self.lowcmd);
        self.writer
            .write(&self.lowcmd)
            .map_err(|e| format!("write: {e}"))?;
        self.poll_state()?;
        self.fill_observation(obs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FL は SDK 3..5、FR は 0..2、RL は 9..11、RR は 6..8 — 1 つでもずれる
    /// と「別の脚が動く」になるので表そのものを固定する。
    #[test]
    fn misa_to_go2_is_the_documented_table() {
        assert_eq!(MISA_TO_GO2[0..3], [3, 4, 5], "FL");
        assert_eq!(MISA_TO_GO2[3..6], [0, 1, 2], "FR");
        assert_eq!(MISA_TO_GO2[6..9], [9, 10, 11], "RL");
        assert_eq!(MISA_TO_GO2[9..12], [6, 7, 8], "RR");
        // 置換であること（重複が無い）
        let mut seen = [false; 12];
        for g in MISA_TO_GO2 {
            assert!(!seen[g]);
            seen[g] = true;
        }
    }

    /// misa-policy-runner の Isaac→Go2 表と、この misa→Go2 表は同じ機体の
    /// 別経路。合成 Isaac→misa が「型順 → 脚順」の置換になっているかを確認
    /// して、2 つの表の食い違いを封じる。
    #[test]
    fn isaac_and_misa_tables_agree_on_the_robot()  {
        use misa_policy_runner::go2::ISAAC_TO_GO2;
        // Isaac index i（型順 FL,FR,RL,RR）→ Go2 → misa 軸
        let go2_to_misa = {
            let mut inv = [0usize; 12];
            for (m, g) in MISA_TO_GO2.iter().enumerate() {
                inv[*g] = m;
            }
            inv
        };
        for isaac in 0..12 {
            let misa = go2_to_misa[ISAAC_TO_GO2[isaac]];
            let (leg, kind) = (misa / 3, misa % 3);
            // Isaac の型順: 0..4 = hip FL,FR,RL,RR / 4..8 = thigh / 8..12 = calf
            assert_eq!(kind, isaac / 4, "isaac {isaac} joint kind");
            assert_eq!(leg, isaac % 4, "isaac {isaac} leg");
        }
    }
}
