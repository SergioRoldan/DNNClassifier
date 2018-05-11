#Only tested in WINDOWS 10 

Pre-requirements:

    - Conda (env python manager): https://conda.io/docs/user-guide/install/index.html
    - Node.js (js framework): https://nodejs.org/es/download/

Requirements:

    Clone the project with the following command:
        $ git clone {TODO}

    Inside the root of the project execute the following:

    Conda:
        - Run the following command to install conda-env:
            $ conda env create -f conda_env/environment.yaml
        - Activate conda-env (default name is sketch-env):
            $ activate sketch-env
        - Install/Check python dependencies:
            $ pip install -r conda_env/packages.txt

    Node:
        - Run the following commands to install node modules:
            $ cd serving_tags_model
            $ npm install

Use the package:

    Inside the root of the project
    Always activate the env before proceding to use the package

    First run the model serving python script:
        $ cd serving_tags_model/model && python modelServing.py
        (Use python serving_tags_model/model/modelServing.py help or python serving_tags_model/model/modelServing.py --help for help)

        Important: To close the serving kill the terminal or ctrl+d while processing a request

    Secondly run the node server handling the requests:
        $ node serving_tags_model/server.js (prefered)
            or 
        $ cd serving_tags_model && npm start
    
    Finally start the client with:
        $ cd client_tags_model && python client.py && cd ..
        (Use python client_tags_model/client.py --help or python client_tags_model/client.py help for help)

        e.g.
            $ cd client_tags_model && python client.py -metrics && cd ..
            $ cd client_tags_model && python client.py -infer --file=JSONs/structured1.json && cd ..

    To create the dataset for the model again run the following python scripts (dataset comes pre-created):
        $ cd tags_model && python datasetCreator.py && cd .. 
        $ xcopy /s "./tags_model/dataset/dictionaries" "./serving_tags_model/model/dataset/dictionaries" /Y
        $ xcopy /s "./tags_model/dataset/tfrecords" "./serving_tags_model/model/dataset/tfrecords" /Y

Useful directories:

    groups_model : Dataset storing & creation. Model creation & training. For groups classification, layer above tag_model
        // In development //

    conda_env : Conda & python environment installation requirements

    client_tags_model : Client to request metrics or inferences to the serving tag model
        client_tags_model/JSONs = JSON sketch input files
        client_tags_model/metrics = Server metrics or inference results

    serving_tags_model : Model serving through a node server at port 7999
        serving_tags_model/model : Model training and serving python script
        serving_tags_model/model/runs : All the training runs with settings, metrics, confusion matrix pic, confusion matrix metrics & log error pic
        serving_tags_model/model/predictions : Tfrecords of the predictions requests
        serving_tags_model/model/dataset : Tfrecords and dictionaries for training and inference

    tags_model : Dataset storing & creation. Model creation & training. For tag classification
        tags_model/runs : All the training runs with settings, metrics, confusion matrix pic, confusion matrix metrics & log error pic
        tags_model/dataset : Dictionaries, JSON sketch input files, symbols, tensorflow training-validation-test records
        tags_model/consumed/symbols : Consumed symbols during dataset creation
        tags_model/documents : Documents for model development (Useless)

    trash: Useless test, files or directories
