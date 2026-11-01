# obstacle lane changing

by mike

## algorithm 

implements test V4 for obstacle detection and then lane changing

1. convert to birds-eye view
2. find objects, lanes
3. lane with highest y-value (so furthest down on image) is one we should switch from
4. don't switch stuff while phases

## data
data is from test run