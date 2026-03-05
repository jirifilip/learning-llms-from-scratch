# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

# %% [markdown]
# ## Simple linear regression
#
# Let's do something like:
#
# $$ child\_height = a * parent\_height + b $$

# %%
n_observations = 200

parent_height = torch.normal(182, 10, size=(n_observations, 1))
intersect = torch.ones_like(parent_height)

feature_matrix = torch.stack(
    [intersect, parent_height], dim=1
).squeeze()

# %%
feature_matrix

# %%
sns.histplot(parent_height).set_title("Parent height")

# %%
error = torch.normal(0, 5, size=(n_observations, 1))

(
    sns
    .histplot(
        error        
    )
    .set_title("Error")
)

# %%
actual_parameters = torch.tensor([-5, 1.1]).unsqueeze(dim=-1)

actual_child_height = feature_matrix @ actual_parameters + error

# %%
actual_child_height.shape, error.shape

# %%
p = sns.scatterplot(
    x=parent_height.reshape(-1).numpy(), 
    y=actual_child_height.reshape(-1).numpy(),
)
p.set_xlabel("parent height")
p.set_ylabel("child height")
p


# %% [markdown]
# ## Make a simple linear model

# %%
class LinearRegression(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        output_size = 1

        # we provide our own bias because I did not remember it can be defined like this
        nn.
        self.layer = nn.Linear(feature_size, output_size, bias=False)

    def forward(self, x: torch.tensor):
        return self.layer(x)


# %%
linear_model = LinearRegression(2)
print(linear_model)

criterion = nn.MSELoss()
# small step because we don't have normalized features
optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.00001)

n_epochs = 20

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    output = linear_model(feature_matrix)
    loss = criterion(output, actual_child_height)
    print(f"{loss = }")
    
    loss.backward()
    optimizer.step()

    print(loss.item())

# %%
linear_model.layer.weight

# %%
with torch.no_grad():
    predicted_child_height = linear_model(feature_matrix)

# %%
sns.scatterplot(
    x=predicted_child_height.reshape(-1).numpy(),
    y=actual_child_height.reshape(-1).numpy()
)
