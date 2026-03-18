import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ObservationPolicyLSTM(torch.nn.Module):
    def __init__(self, num_obs, num_actions, embedding_dim=32, hidden_dim=64):
        super(ObservationPolicyLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        # 1. THE TRANSLATOR (Embedding)
        # Neural networks hate integers (0, 1, 2). They love vectors.
        # This layer turns obs ID '3' into a vector like [0.1, -0.5, 0.0, ...]
        self.embedding = torch.nn.Embedding(num_obs, embedding_dim)

        # 2. THE BRAIN (LSTM Cell)
        # This is the recurrent core.
        # It takes an input vector (size 32) and the old memory (size 64).
        # It outputs the new memory (size 64).
        self.lstm = torch.nn.LSTMCell(embedding_dim, hidden_dim)

        # 3. THE DECIDER (Linear/Fully Connected)
        # This takes the memory (size 64) and decides the action scores.
        # Output size = num_actions
        self.fc = torch.nn.Linear(hidden_dim, num_actions)

    def forward(self, obs_idx, hidden_state):
        # Step 1: Translate Integer -> Vector
        # obs_idx shape: [Batch_Size] -> embedded shape: [Batch, 32]
        embedded = self.embedding(obs_idx)

        # Step 2: Update Memory
        # Input: (New Vector, Old Memory Tuple)
        # Output: (New Hidden h_x, New Cell State c_x)
        h_x, c_x = self.lstm(embedded, hidden_state)

        # Step 3: Decide Action
        # Input: New Hidden State h_x
        # Output: Raw scores for actions (Logits)
        logits = self.fc(h_x)

        # Return the scores AND the new memory (to be used in the next loop iteration)
        return logits, (h_x, c_x)

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))
