#### This repository is the implementation of Meta-LSTM in "Meta Multi-Task Learning for Sequence Modeling." AAAI-18 https://arxiv.org/abs/1802.08969

##### Dependencies

TensorFlow:  == 1.0.1

##### How to use

```python
meta_cell = rnn_cell.BasicLSTMCell(meta_cell_unit_nums, state_is_tuple=False)
lstm_cell = MetaLSTMCell(unit_nums, meta_cell)
outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, ph_seqLen, 
                           dtype=tf.float32 ,swap_memory=True, scope = 'meta-lstm-')
```

##### In Multi-Task Learning

init a meta_cell with a shared scope, this work can be solved.  Pay attention to the  `reuse_variables()`

#### In addition the version implemented with PyTorch may be updated later.