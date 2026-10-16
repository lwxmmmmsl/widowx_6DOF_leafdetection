"""将真机采样映射为 PhysX 可接受的启动状态，不改写真机配置或 URDF。

编码器读数可能超出 URDF 的软件限位，例如 sleep 附近的肩关节。
clamp 模式仅调整仿真初值；strict 模式要求采样值本身合法。
角度关节单位为 rad，手指滑动关节单位为 m。
"""
import math


def prepare_initial_joints(measured, limits, policy='clamp'):
    """返回独立的仿真初值字典和可审计的调整记录。

    Args:
        measured: hardware.yaml 中的原始关节位置，不会原地修改。
        limits: 从原 URDF 读取的关节限位，包括连续旋转的 gripper。
        policy: clamp 允许投影越界值；strict 对实际越界立即报错。

    有限关节边界内缩极小距离，避免 USD 角度转换、float32 舍入后再次
    超限。合法但恰好在边界上的值也会留下数值内缩记录。
    """
    if policy not in ('clamp', 'strict'):
        raise ValueError(f'Unknown initial_joint_policy: {policy!r}')
    initial = dict(measured)
    adjustments = []
    for name, value in measured.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite initial position: {name}={value}')
        if name not in limits:
            raise ValueError(f'Unknown initial joint in profile: {name}')
        limit = limits[name]
        # 连续关节没有上下界，不应误当成零范围关节裁剪。
        if 'lower' not in limit or 'upper' not in limit:
            initial[name] = value
            continue
        lower, upper = limit['lower'], limit['upper']
        if not lower < upper:
            raise ValueError(f'Invalid URDF limits for {name}: [{lower}, {upper}]')
        outside = value < lower or value > upper
        if outside and policy == 'strict':
            raise ValueError(
                f'Initial joint {name}={value:.9f} outside URDF limits '
                f'[{lower:.9f}, {upper:.9f}]. Re-capture a valid real pose or '
                'set initial_joint_policy: clamp in scene.yaml for simulation only.'
            )
        margin = 1e-6 if name in ('left_finger', 'right_finger') else 1e-5
        margin = min(margin, (upper-lower)/4)
        projected = min(max(value, lower+margin), upper-margin)
        initial[name] = projected
        if projected != value:
            adjustments.append({
                'joint': name,
                'measured': value,
                'simulated': projected,
                'lower': lower,
                'upper': upper,
                'unit': 'm' if name in ('left_finger', 'right_finger') else 'rad',
                'reason': 'outside_urdf_limits' if outside else 'float32_boundary_margin',
            })
    return initial, adjustments
