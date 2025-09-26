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
    function startAllRecording(){
        console.log('Command received: Start all recordings');
        const recordingsDir = path.join(__dirname, 'recordings');
        if (!fs.existsSync(recordingsDir)){
            fs.mkdirSync(recordingsDir);
        }   
        clients.forEach((client, clientId) => {
            if(!client.isRecording){
                const filename = '${clientId}_${Date.now()}.webm';
                client.filestream = fs.createWriteStream(path.join(recordingsDir, filename));
                console.log(`New recording file created: ${filename}`);
                client.isRecording = true;
                client.socket.send(JSON.stringify({command: 'start_recording'}));
    }});
        sendHostUpdate();
    }

    function stopAllRecording(){
        console.log('Command received: Stop all recordings');
        clients.forEach((client, clientId) => {
            if(client.isRecording){
                client.socket.send(JSON.stringify({command: 'stop_recording'}));
     } });
    }
    function sendHostUpdate(){
        if(!hostSocket) return
            const clientList = Array.from(clients.keys()).map(clientId => ({
                clientId,
                isRecording: clients.get(clientId).isRecording
            }));
        const isRecordingAll = clientList.length>0 && clientList.every(client => client.isRecording);
        hostSocket.send(JSON.stringify({
            type : 'state_update',
            isRecordingAll,
            clients:clientList
        }));
    }
    
    function getLocalIP(){
        const interfaces = os.networkInterfaces();
        for(const name of Object.keys(interfaces) ){
            for(const iface of interfaces[name]){
                if(iface.family === 'IPv4'  && !iface.internal) return iface.address;
            }
            }
        return 'localhost';
    }

    const PORT = 8000;

    server.listen(PORT, () => {
        const ip = getLocalIP();
        console.log(`Websocket server running on port ${PORT}`);
        console.log(`ws://${ip}:${PORT}`);
    }
    );