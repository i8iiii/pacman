

CSC14003 | Introduction to Artificial IntelligenceLab
## Lab
Hide and Seek Arena
## 1    Introduction
Hide and Seek Arena aims to help students understand,  implement,  and optimize search al-
gorithms through an automated arena that simulates the game of hide-and-seek,  where student
teams can compete against one another for ranking.
Each student team will develop both of the following agents:
-  Hide agent:  Hide for as long as possible.
-  Seek agent:  Find and catch as quickly as possible.
The agents will be automatically matched in the Arena to compute win rates and rankings.
## 2    Arena
The Arena is an automated framework,  implemented in Python,  that serves as the competitive
environment for the agents.  Given the teams’ submissions, the Arena will (1) automatically import
each team’s agents; (2) run a series of matches between teams; and (3) output the results.
The model of the Arena is illustrated as below:
## Algorithm 1 Agent – Environment Interaction Loop
while episode not terminated do
a
p
← PacmanAgent.step()
a
g
← GhostAgent.step()
## Environment.step(a
p
, a
g
## )
Update agent positions
Check collisions or termination
s← observe new environment state
end while
At each step in Algorithm 1, the provided information includes:
-  A 21× 21 map (where 0 indicates a traversable cell and 1 indicates a non-traversable cell).
-  The current positions of the two agents.
Faculty of Information Technology, University of Science, VNU-HCMPage 1

CSC14003 | Introduction to Artificial IntelligenceLab
## # # # # # # # # # # # # # # # # # # # # #
## # . . . . . . . . . # . . . . . . . . . #
## # . # # # . # # # . # . # # # . # # # . #
## # . # # # . # # # . # . # # # . # # # . #
## # . . . . . . . . . . . . . . . . . . . #
## # . # # # . # . # # # # # . # . # # # . #
## # . . . . . # . . . # . . . # . . . . . #
## # # # # # . # # # . # . # # # . # # # # #
## # # # # # . # . . . G . . . # . # # # # #
## # # # # # . # . # # . # # . # . # # # # #
## # . . . . . . . # . . . # . . . . . . . #
## # # # # # . # . # # # # # . # . # # # # #
## # # # # # . # . . . . . . . # . # # # # #
## # # # # # . # . # # # # # . # . # # # # #
## # . . . . . . . . . # . . . . . . . . . #
## # . # # # . # # # . # . # # # . # # # . #
## # . . . # . . . . . P . . . . . # . . . #
## # # # . # . # . # # # # # . # . # . # # #
## # . . . . . # . . . # . . . # . . . . . #
## # . # # # # # # # . # . # # # # # # # . #
## # . . . . . . . . . . . . . . . . . . . #
## # # # # # # # # # # # # # # # # # # # # #
Figure 1:  Arena map (P: Pacman, G: Ghost, #:  Walls, .:  Empty path)
Faculty of Information Technology, University of Science, VNU-HCMPage 2

CSC14003 | Introduction to Artificial IntelligenceLab
-  The current step count.
At each step, both agents receive the same state information and make their decisions simultane-
ously.
The win conditions are defined as follows:
-  The Seek agent wins:  if the Seek agent touches the Hide agent (Manhattan distance < 2).
-  The Hide agent wins:  if the maximum number of allowed steps is reached.
Please  note  that,  the  framework  is  provided  to  you  for  reference,  evaluting  your  own  agent
against the above conditions may require some tweakings.  Also note that the framework have a
”fog” setting which we will NOT use in this lab.
The source code of this framework is provided as pacman.zip in the Google Drive folder.  The
source code will be provided to students with the following directory structure:
## Arena/
src/ ...................................... Framework source code (do NOT modify)
submissions/ .................................................. Student workspace
examplestudent/
agent.py .............................................Reference implementation
studentid/
agent.py .............................................. Student team’s agent
STUDENTGUIDE.md ............................................... Student instructions
rungame.sh .........................................................Quick-run script
Besides  that,  the  given  folder  also  includes  an Clear  folder  featuring  selected  instances  of
previous courses where the top submissions are sampled.  About the speed, if both agents move
at the same speed, the Seek Agent can never win because it cannot make contact with the Hide
Agent.  To ensure the fairness, the Seek Agent moves at a speed of 2 cells per step when moving
in straight line but cannot move with an L-shaped immediately during a turn.
At each step, a student team’s agent must return an action in the following format:
1 class  AgentInterface(ABC):
## 2      @abstractmethod
3      def  step(self , map_state , my_position , enemy_position , step_number):
4           # For  Pacman  Agent: return a Move or (Move , steps), where:
5           # - steps  is an int  from 1 to the  configured  max  straight -line  speed.
6           # For  Ghost  Agent: return a Move (UP , DOWN , LEFT , RIGHT , or STAY).
Faculty of Information Technology, University of Science, VNU-HCMPage 3

CSC14003 | Introduction to Artificial IntelligenceLab
The recommended algorithms are: Minimax, Monte Carlo, Alpha-beta Pruning, Expecti-minimax,
## .etc.
## 3    Tournament
Each team will participate in both roles, Hide and Seek, and will compete against the agents of
all other teams.  Based on the outcomes, the win rates are computed as:
winrate
hide
## =
Hide wins
Hide games
win
rate
seek
## =
Seek wins
Seek games
The results are recorded in the following example table.
Table 1:  Match Results Between Teams (Hide vs.  Seek)
Hide \ SeekTeam 1Team 2Team 3Team 4
## Team 1–011
## Team 20–10
## Team 310–1
## Team 4001–
Here, a value of 0 denotes a Hide win, while a value of 1 denotes a Seek win.
Based on Table 1, an example of the win rates for Team 1 is as follows:
-  As the Hide agent, Team 1 wins 1 out of 3 matches, resulting in winrate
hide
## = 33.3%.
-  As the Seek agent, Team 1 wins 2 out of 3 matches, resulting in winrate
seek
## = 66.6%.
Note that,  besides raw winning result,  we also record your agent’s average steps required to
end each match.  For Pacman,  you should also minimize this value while for Ghost,  you should
also maximize this value.  In order to tie-breaking the teams with similar winning rate, we evaluate
their differences between Pacman’s and Ghost’s average steps where one with lower difference ranks
higher.
The scoring criteria for each team is defined as follows:
CriterionScore
Algorithm implementation completeness3
Ranking in the initial submission (see timeline in Section 4)Up to 3
Ranking in the optimized submission (see timeline in Section 4)Up to 4
Faculty of Information Technology, University of Science, VNU-HCMPage 4

CSC14003 | Introduction to Artificial IntelligenceLab
## 4    Submission
Each team must submit a single groupid.zip file to Moodle containing the following directory
structure where groupid denotes a single integer for your group’s ID only, for example please use
1.zip INSTEAD of group1.zip or group01.zip.  The general structure of your submitted zip
folder should be:
groupid/
agent.py
etc.
The submitted agent is allowed to import other files located within the same submission directory.
The submission timeline is specified as follows:
## 5    Notices
Please pay attention to the following notices:
-  This is a GROUP assignment.
-  Late submissions will not be accepted under any circumstances.
-  This project has two submission phases.  Only teams that submitted in the first phase are
allowed to submit in the second phase.
-  Student teams must ensure that their agent returns an action within a maximum of 1 second
while consuming at most 128 MB under the setting of Google Colab’s CPU-only instance.
The built-in time library may be used together with all built-in libraries bundled with Python
-  External libraries available for your team includes:  numpy, pandas, scipy, and gurobi.
-  AI tools are NOT restricted; however, students should use them wisely.  Lab instructors
have the right to conduct additional oral interviews to assess their knowledge of the project.
-  Any form of plagiarism, dishonesty, or misconduct will result in a grade of zero for the course.
The end.
Faculty of Information Technology, University of Science, VNU-HCMPage 5