Active Sample Selection For Matrix Completion
=======


This contains the code used to simulate active sample selection on various datasets, all included in the Matlab/data folder.

Report
--------

The full report can be found in the Report directory under [thesis.pdf](http://sg3510.github.io/al_proj/Report/thesis.pdf "Project Final Report")

Aim of Project
--------
This project builds an top of a [recommender system](http://en.wikipedia.org/wiki/Recommender_system "Wikipedia - Recommender System") for the purpose of [active learning](http://en.wikipedia.org/wiki/Active_learning_(machine_learning) "Wikipedia - Active Learning"). 

An example of a use case is to assume we have a user-movie database in matrix form, with many incomplete entries. From the current data set we may know that users liking Star Wars 1 are very likely to like Star Wars 2. If we are given the opportunity to ask any user his opinion on a film, asking a user his opinion will on Star Wars 1 or 2 may not be very useful in terms of data prediction. Instead it may be worth asking a user his opinion on Pulp Fiction. This corresponds to an empty row-colum entry in the origial dataset. The aim of this project is thus to determine mathematical criteria which is useful in locating the best row-colum coordinate to query when allowed to build up the size of the dataset.

References
--------
References for non-original code is found in the report references. 
