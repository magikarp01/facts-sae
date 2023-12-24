setup:
- git clone https://github.com/saprmarks/dictionary_learning.git (actually don't do this)
- git clone https://github.com/magikarp01/tasks.git

dictionary_learning directory is from saprmarks, but with some alterations 

use same requirements as mechanistic-unlearning, along with a few more here

srun --nodes=1 --cpus-per-task=4 --gres=gpu:1 --time=2:00:00 --pty bash -i

srun --nodes=1 --cpus-per-task=4 --time=2:00:00 --pty bash -i