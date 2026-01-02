import numpy as np

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)
    
    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            total = 0
            
            # For absorbing state, value should remain 0
            if s == env.absorbing_state:
                continue
            
            # Get action from policy
            a = policy[s]
            
            # Bellman expectation equation
            for ns in range(env.n_states):
                prob = env.p(ns, s, a)
                reward = env.r(ns, s, a)
                total += prob * (reward + gamma * value[ns])
            
            value[s] = total
            delta = max(delta, abs(v - value[s]))
        
        if delta < theta:
            break
            
    return value

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    for s in range(env.n_states):
        # Absorbing state - any action is fine (won't be used)
        if s == env.absorbing_state:
            policy[s] = 0
            continue
            
        best_action = 0
        best_value = float('-inf')
        
        # Try all actions
        for a in range(env.n_actions):
            total = 0
            for ns in range(env.n_states):
                prob = env.p(ns, s, a)
                reward = env.r(ns, s, a)
                total += prob * (reward + gamma * value[ns])
            
            # Choose action with highest value
            if total > best_value:
                best_value = total
                best_action = a
        
        policy[s] = best_action
        
    return policy

def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    for i in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        new_policy = policy_improvement(env, value, gamma)
        
        if np.array_equal(policy, new_policy):
            break
            
        policy = new_policy
    
    return policy, value

def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=float)
    
    for i in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            # Skip absorbing state - value should remain 0
            if s == env.absorbing_state:
                continue
                
            v = value[s]
            best_value = float('-inf')
            
            # Find best action value
            for a in range(env.n_actions):
                total = 0
                for ns in range(env.n_states):
                    prob = env.p(ns, s, a)
                    reward = env.r(ns, s, a)
                    total += prob * (reward + gamma * value[ns])
                
                if total > best_value:
                    best_value = total
            
            value[s] = best_value
            delta = max(delta, abs(v - value[s]))
        
        if delta < theta:
            break
    
    # Extract policy from final value function
    policy = policy_improvement(env, value, gamma)
    return policy, value