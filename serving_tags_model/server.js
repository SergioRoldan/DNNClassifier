var express = require('express');
var bodyParser = require('body-parser');
var timeout = require('connect-timeout');
var _ = require('underscore');

var app = express();
var PORT = process.env.PORT || 7979;

app.use(timeout(45000));
app.use(bodyParser.json({ limit: '10mb' }));
app.use(haltOnTimedout);

function haltOnTimedout(req, res, next) {
    if (!req.timedout) next();
    else res.status(500).send();
}

var zmq = require('zeromq');

app.post('/infer', function(req, res) {
    console.time('Infer')
    console.log('Request');
    console.log(req.headers);
    
    let requester = zmq.socket('req');
    let port_sock = '7999';
    requester.connect('tcp://localhost:' + port_sock);

    let body = _.pick(req.body, 'inferences')
    let inferences = body.inferences

    if(inferences.length == 0 || typeof  inferences === 'undefined')
        return res.status(400).send();

    let client_id = req.hostname + req.ip;
    requester.send("Request inference from client " + client_id);
    console.log("Message send");

    requester.on('message', function (reply) {
        console.log("Message received");
        let socket = zmq.socket('req');
        let new_port = reply.toString().split(': ')[1];

        if (new_port.length == 0 || typeof new_port === 'undefined') 
            return res.status(500).send();

        socket.connect('tcp://localhost:' + new_port);
        socket.send('Ping from client ' + client_id);

        socket.on('message', function (reply) {

            let msg = reply.toString();

            if (msg.indexOf('ping') !== -1) {
                console.log('Reply', msg);
                socket.send('Inference from client ' + client_id + '> ' + JSON.stringify(inferences))
            } else if (msg.indexOf('inference') !== -1) {
                try {
                    console.log('Reply', msg.split(': ')[0]);
                    let results = msg.split(': ')[1].replace('[', '').replace(']', '');
                    results = results.split('), (');

                    for (let i = 0; i < results.length; i++) {
                        if (results[i].indexOf('(') !== -1)
                            results[i] = results[i].replace('(', '');
                        if (results[i].indexOf(')') !== -1)
                            results[i] = results[i].replace(')', '');
                        if (results[i].indexOf('\'') !== -1)
                            results[i] = results[i].replace(new RegExp('\'', 'g'), '');
                    }

                    for (let i = 0; i < results.length; i++) {
                        let tmp = results[i].split(', ');
                        results[i] = {
                            'Object': {
                                'name': inferences[i]['name'],
                                'identifier': inferences[i]['identifier']
                            },
                            'Tag estimated': tmp[0],
                            'Estimation confidence': tmp[1]
                        };
                    }

                    let count = 0;
                    for (let i = 0; i < results.length; i++)
                        if (parseFloat(results[i]['Estimation confidence']) < 80) {
                            count++;
                            //console.log(results[i]);
                        }

                    console.log('Total: ' + results.length + '\nNot confident: ' + count)

                    socket.send('Ack & close from client ' + client_id);
                    socket.close();

                    return res.json(JSON.stringify(results));
                } catch (error) {
                    console.log('Error '+error.toString()+' handling the response');
                    return res.status(500).send('Exception ' + error.toStrin() + 'thrown during response handle. Try again later')
                }
                
            } else if (msg.indexOf('exception') !== -1) {
                e = msg.split('<', 1)[1].split('>', 1)[0]
                console.log('Exception thrown during inference: '+ e);
                return res.send('Exception '+ e+ 'thrown during inference. Try again later')
            }
        });

    });

});

app.get('/metrics', function(req, res) {

    console.log('Request');
    console.log(req.headers);

    let requester = zmq.socket('req');
    let port_sock = '7999';
    requester.connect('tcp://localhost:' + port_sock);

    let client_id = req.hostname + req.ip;
    requester.send("Request metrics from client " + client_id);

    console.log("Message send");

    requester.on('message', function (reply) {
        console.log("Message received");
        rep = reply.toString();
        return res.send(rep);
    });

});

app.listen(PORT, function () {
    console.log('Express listening on port ' + PORT);
});