"""构造完整的 URDF 导入配置，避免隐式加载 Isaac/Omni。

传入已导入的配置类，满足 AppLauncher 必须先启动的初始化顺序。

Build importer settings without loading Isaac before AppLauncher."""


def robot_import_config(config_type, asset_path, usd_dir):
    """Explicitly satisfy Isaac Lab's mandatory joint-drive gain schema.

    Imported drives start at zero gains. Articulation's ImplicitActuatorCfg
    applies the configured arm/finger PD gains before physics execution.
    """
    # import.urdf 已移除 mimic 标签：左右指由仿真明确发对称位置指令。
    # 本机 importer 的 parse_mimic 开启时会生成从动关节并扩大其限位，
    # 因此不能依赖该版本名称相反的 convert_mimic 开关。
    return config_type(
        asset_path=str(asset_path),
        usd_dir=str(usd_dir),
        usd_file_name='wx250s.usd',
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        collider_type='convex_decomposition',
        self_collision=False,
        joint_drive=config_type.JointDriveCfg(
            gains=config_type.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
