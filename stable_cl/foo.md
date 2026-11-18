$$
A_{t,j}
$$

$$
j \le t
$$

$$
2t + 2
$$


$$
t, j = 0..4
$$

$$
\text{acc} = \tfrac15 \sum_{j} A_{4,j}, \qquad
    \text{forgetting} = \tfrac14 \sum_{j<4} \Bigl(\max_{j \le t \le 3} A_{t,j} - A_{4,j}\Bigr),
$$

$$
\text{plasticity} = \tfrac15 \sum_t A_{t,t}, \qquad
\text{newest} = A_{4,4}, \qquad
\text{old} = \tfrac14 \sum_{j<4} A_{4,j}.
$$
