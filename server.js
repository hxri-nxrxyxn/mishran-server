const http = require('http');
const {WebSocketServer}= require('ws');
const os = require('os');
const fs = require('fs');
const path = require('path');

const server = http.createServer();
const wss = new WebSocketServer({server});

let clients = new Map();
let hostSocket = null;

wss.on('connection', (socket, req) => {
    const clientId = req.url.split('/').pop();
    if (clientId === 'host_monitor') {
        hostSocket = socket;
        console.log('Host monitor connected');
        sendHostUpdate();
        
        socket.on('message', (message) => {
            const data = JSON.parse(message);
            if (data.command === 'start_all') startAllRecording();
            if (data.command === 'stop_all') stopAllRecording();
        });
        
        socket.on('close',() => {
            hostSocket = null;
            console.log('Host monitor disconnected');
        });
    } else {
        console.log(`Client ${clientId} connected`);
        clients.set(clientId, {socket, isRecording: false, filesream: null});
        sendHostUpdate();

        socket.on('message', (message) => {
            const client= clients.get(clientId);
            try{
                const data = JSON.parse(message);
                if (data.type === 'recording_fully_stopped'){
                    console.log("Client ${clientId} confirmed recording fully stopped");
                };
                client.isRecording = false;

                if(client.filesream) {
                    client.filesream.end();
                    client.filesream = null;
                    console.log(`Recording file saved for client ${clientId}`);
                }
                sendHostUpdate();
                return;
            } catch (error) {
                if(client?.isRecording && client.filesream) {
                    client.filesream.write(message);
                }
            }
        });
            socket.on('close',() => {
                console.log(`Client ${clientId} disconnected`);
                const client = clients.get(clientId);
                if(client?.filestream){
                    client.filestream.end();
                }
                clients.delete(clientId);
                sendHostUpdate();
            });
            } 
            
        });

