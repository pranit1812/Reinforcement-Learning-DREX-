    # fragments, preferences = drex_trainer.generate_fragments_and_preferences(drex_trainer.ranked_trajectories, 100, 50, 0.3)
    # for i in range(len(fragments)):
    #     fragment = fragments[i]
    #     idx_a, idx_b = 0, 1
    #     reward_1, reward_2 = np.sum(fragment[idx_a].rews), np.sum(fragment[idx_b].rews)
    #     gt_prob = 1/(1+np.exp(reward_2-reward_1))
    #     gt_loss = -np.log(gt_prob)
    #     predicted_reward_1 = []
    #     for i in range(fragment[idx_a].acts.shape[0]):
    #         obs, act, next_obs, done = fragment[idx_a].obs[i][None,:], fragment[idx_a].acts[i][None,:], fragment[idx_a].obs[i+1][None,:], np.array([False])
    #         predicted_reward_1.append(reward_net.predict(obs, act, next_obs, done))
    #     predicted_reward_1 = np.sum(predicted_reward_1)
    #     predicted_reward_2 = []
    #     for i in range(fragment[idx_b].acts.shape[0]):
    #         obs, act, next_obs, done = fragment[idx_b].obs[i][None,:], fragment[idx_b].acts[i][None,:], fragment[idx_b].obs[i+1][None,:], np.array([False])
    #         predicted_reward_2.append(reward_net.predict(obs, act, next_obs, done))
    #     predicted_reward_2 = np.sum(predicted_reward_2)
    #     pred_prob = 1/(1+np.exp(predicted_reward_2-predicted_reward_1))
    #     pred_loss = -np.log(pred_prob)
    #     # print(f"GT: {gt_prob}, Predicted: {pred_prob}")
    #     print(f"True reward: {(reward_1, reward_2)}, Predicted reward: {(predicted_reward_1, predicted_reward_2)}, GT: {gt_prob}, Predicted: {pred_prob}")
    
    # print('---------------------------------------------------------------------------------------------------')