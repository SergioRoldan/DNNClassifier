import parseJSON as pJ
import datasetWriter as dW

train_percent = 55
val_percent = 30
tag_non_erase = 50

print('\nDataset Creation Started\n')
print('\n-- Extracting & parsing JSONs...\n')
pJ.run()
print('\n-- Writing TF records...\n')
dW.run(train_percent, val_percent, tag_non_erase)
print('\nDataset Creation Finished\n')