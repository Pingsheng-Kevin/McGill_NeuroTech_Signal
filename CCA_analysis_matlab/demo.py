import matlab.engine

eng = matlab.engine.start_matlab()
dummy = eng.rand(5, 8, 2, 2)
dummy_template = eng.rand(5, 8, 2, 2)
print(dummy)
print(dummy[1][1][:][:])
print(eng.FBCCA_IT(dummy, 8.0, 8.0, 88.0, 4.0, dummy_template, 1.0, 0.0, 2.0, 250.0, 2.0))
