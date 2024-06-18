The Taxi Problem

### Description
There is an 5*5 grid, whose boundary is enclosed by walls. There are some walls between east-west adjacent cells.
When the episode starts, the taxi starts off at a random cell and the passenger is at a random cell. The taxi drives to the passenger's cell, picks up the passenger, drives to the passenger's destination, and then drops off the passenger. Once the passenger is dropped off, the episode ends.
The wall is denoted as |.
The taxi is denoted as T, the passenger is denoted by P and the destination is denoted by D.
If the taxi is at the passenger's location, this location is denoted by O.
If the taxi has picked up the passenger, their common location is denoted by X.

### Rewards
- -1 per step unless other reward is triggered.
- +20 delivering passenger.
- -10 executing "pickup" and "drop-off" actions illegally.
