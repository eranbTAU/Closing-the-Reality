
'''
reward function which was used in the paper.
'''

def rew1(diag_mm, divider):
    '''
    :param diag_mm: maximum distance in [mm].
    :param divider: sacledown factor for reward.
    :return: reward
    '''
    def f(action, agent_pose, other_dist, landmark_dist, other_rel_dir,
          landmark_rel_dir, other_rel_orient, landmarks_rel_orient):
        # continuous distance attraction to target, penalty for collision.
        rew = 0
        for d in landmark_dist:
            if d < 150:
                rew += -5 # target collision
            else:
                rew += -20 * d / diag_mm # target attraction
        for d in other_dist:
            if d < 150:
                rew += -5 # neighbours collision
        return rew / divider
    return f


def get_reward_func(name, diag_mm, divider):
    return globals()[name](diag_mm, divider)
