 class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn(env, episodes=500, batch_size=32, gamma=0.9, lr=0.001):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    memory = deque(maxlen=1000)

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0

        for t in range(50):  # Limit steps per episode
            if random.random() < 0.1:  # Exploration
                action = env.action_space.sample()
            else:  # Exploitation
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            memory.append((state, action, reward, next_state))
            state = next_state
            total_reward += reward

            if done:
                break

        # Training step
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            for state, action, reward, next_state in batch:
                target = reward + gamma * torch.max(model(next_state)).item()
                predicted = model(state)[0, action]

                loss = criterion(predicted, torch.tensor(target))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward}")

    torch.save(model.state_dict(), "dqn_webcrawler.pth")
    print("DQN model saved.")

# Initialize the environment and train
env = WebCrawlerEnv(sites)
train_dqn(env)
