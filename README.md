# Master Project -- ML applied to HHCRSP with paGOMEA
Outline of scripts:
•	generate.py is the starting point: calls modules relevant to create (instance.py) and solve (gomea.py) an instance in order to generate trainable data.
•	instance.py includes methods to generate a random problem instance
•	gomea.py includes methods needed to call paGOMEA: increase the population size to get a higher solution quality (but also computation time), by default population = 200 works well on instances with up to approx. 100 jobs.
•	measures.py contains methods needed by gomea.py
•	schedule.py contains methods to store, create and/or generate a schedule (e.g., a solution of paGOMEA)
•	test_env.py includes script that trains ML models on data set obtained from generate.py
•	inspect_solutions.py analyzes how solutions to the HHCRSP look like
•	evaluate_routes.py evaluates the objective function of a solution
•	For the computation of features:
o	akkerman_functions.py includes the computation of features from the scientific paper of Akkerman
o	point_pattern_features.py computes features based on the paper of Mei
o	new_feature_functions.py computes some of my own features
•	DTRP_chapter.py includes the simulation of chapter 5, not related to anything earlier.
