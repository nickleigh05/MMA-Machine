import os
import time
import mujoco


def compose_mma_scene(arena_path: str, humanoid_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Merge two prefixed humanoids into the arena. Returns (model, data)."""
    spec = mujoco.MjSpec.from_file(arena_path)
    humanoid_a = mujoco.MjSpec.from_file(humanoid_path)
    humanoid_b = mujoco.MjSpec.from_file(humanoid_path)

    # MjSpec.attach(child, prefix, suffix, frame=frame) keeps freejoint at top level.

    ### Fighter A: +1.5m on X, 90° around Z → faces -X (into centre) ###

    fa = spec.worldbody.add_frame()
    fa.pos = [1.5, 0.0, 0.0]
    fa.quat = [0.7071068, 0.0, 0.0, 0.7071068]
    spec.attach(humanoid_a, 'fighter_a_', '', frame=fa)

    ### Fighter B: -1.5m on X, -90° around Z → faces +X (into centre) ###

    fb = spec.worldbody.add_frame()
    fb.pos = [-1.5, 0.0, 0.0]
    fb.quat = [0.7071068, 0.0, 0.0, -0.7071068]
    spec.attach(humanoid_b, 'fighter_b_', '', frame=fb)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


if __name__ == '__main__':
    _dir = os.path.dirname(os.path.abspath(__file__))
    m, d = compose_mma_scene(
        os.path.join(_dir, 'arena_octagon.xml'),
        os.path.join(_dir, 'humanoid_base.xml'),
    )

    print(f'nq={m.nq}  nu={m.nu}  nsensor={m.nsensor}  nbody={m.nbody}')
    assert m.nq == 80,  f'nq expected 80, got {m.nq}'
    assert m.nu == 66,  f'nu expected 66, got {m.nu}'
    print('PASS — launching viewer')

    ### Apply keyframe 0 so both fighters start in standing pose. freejoints manually. ###

    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[0]  = 1.5   # fighter_a X
    d.qpos[40] = -1.5  # fighter_b X
    mujoco.mj_forward(m, d)

    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.distance = 30.0
        v.cam.elevation = -30.0
        v.cam.azimuth = 90.0
        v.cam.lookat[:] = [0.0, 0.0, 1.0]
        v.sync()
        while v.is_running():
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(m.opt.timestep)
