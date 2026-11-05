from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from g1d_config import G1D_USD, HOME_ROOT_Z, WHEEL_JOINTS
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
class S(InteractiveSceneCfg):
    robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot', spawn=sim_utils.UsdFileCfg(usd_path=G1D_USD), init_state=ArticulationCfg.InitialStateCfg(pos=(0.0,0.0,HOME_ROOT_Z)), actuators={'a': ImplicitActuatorCfg(joint_names_expr=['.*'])})
import isaaclab.sim as ss
sim = ss.SimulationContext(ss.SimulationCfg(dt=1/60, device='cpu'))
sc = InteractiveScene(S(num_envs=1, env_spacing=0.0))
r = sc['robot']
fields = [f for f in dir(r.data) if not f.startswith('_')]
print('AVAIL robot:', sorted(fields), flush=True)
app.close()
