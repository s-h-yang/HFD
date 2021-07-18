Three files for a node-labeled hypergraph.

-------
node-labels-$dataset.txt
-------

The number of line i corresponds to the label for node i.


-------
label-names-$dataset.txt
-------

Map from label indices to label names. Line i gives the name for label i.


-------
hyperedges-$dataset.txt
-------

Each line represents one hyperedge, given by a comma-separated list of node indices.




-------
Dataset construction
-------

The raw data is from the Trivago RecSys Challenge 2019:
https://recsys.acm.org/recsys19/challenge/

The hypergraph is constructed from the training data. Each row of the original
training dataset file represents an action performed by a user during a browsing
session on Trivago. The row includes an id for the browsing session, an
explanation for the type of action a user performed, an id for the accommodation
the user interacted with, and a location for that accommodation.

Nodes in the hypergraph represent accommodations, and hyperedges are sets of
accommodations for which a user performed the "click-out" action during the same
session, which means the user was forwarded to a partner site. In constructing
the hypergraph we additionally applied the following steps:
* We removed accommodations for which no metadata is known.
* If a user "clicks-out" in the same accommodation multiple times in the same session, the accommodation only appears once in the hyperedge
* We take only the largest connected component of the hypergraph, discarding smaller components
* We then remove all hyperedges of size 1
The resulting hypergraph has 172738 nodes (all of which have degree at least 1) and 233202 hyperedges.
