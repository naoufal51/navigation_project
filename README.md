# Navigation

Train an agent to navigate within a large environment (similar to Banana Collector environment) using advanced Deep Q-Network (DQN) techniques.

## 🛠 Setup & Dependencies

1. **Python 3.8 Environment**: 

   - **Linux/Mac**: 
     ```bash
     conda create --name drlnd python=3.8
     source activate drlnd
     ```
   - **Windows**: 
     ```bash
     conda create --name drlnd python=3.8
     activate drlnd
     ```

2. **Repository Setup**:
   
   ```bash
   git clone https://github.com/naoufal51/navigation_project.git
   cd navigation_project/python
   pip install .
   ```

3. **IPython Kernel Configuration**:
   
   ```bash
   python -m ipykernel install --user --name drlnd --display-name "drlnd"
   ```

4. **Environment Download**: Depending on your OS, download and place in the `p1_navigation` folder:
   
    - **Linux**: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - **Mac OSX**: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - **Windows 32-bit**: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - **Windows 64-bit**: [Download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

5. **Environment Path Configuration**: In `p1_navigation/train.py`, update the environment path:
   
   ```python
   with BananaUnityEnv(file_name=<Your_Path_Here>, seed=seed, no_graphics=False) as env:
   ```

## 📂 Project Structure

```
.
├── LICENSE.md
├── README.md
├── p1_navigation
│   ├── Banana.app                         # Unity training environment.
│   ├── agent.py                           # Agent logic implementing DQN.
│   ├── env.py                             # Environment handler.
│   ├── figures                            # Visualizations of training outcomes.
│   ├── main_runner.py                     # Script to initiate agent training.
│   ├── model.py                           # DQN neural network structures.
│   ├── models                             # Checkpoints and resultant models.
│   └── test.py                            # Evaluate trained agents.
└── python                                 # Installation scripts for dependencies.
```

## 🚀 Usage

Kickstart agent training with:

```bash
cd p1_navigation
python3 main_runner.py
```

Utilize configurations in `main_runner.py` for tuning.

## 💡 Contributions

Contributions are welcome! Please fork this repository, make your enhancements, and initiate a pull request.

## 🙌 Acknowledgement
[Udacity](https://www.udacity.com/) for providing the starter code and training environment.