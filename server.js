const http = require('http');
const { WebSocketServer } = require('ws');
const os = require('os');
const fs = require('fs');
const path = require('path');

const server = http.createServer();
const wss = new WebSocketServer({ server });

let clients = new Map();
let hostSocket = null; // Will store { socket, filestream }
let currentRecordingSessionPath = null; // Path for the current session's files

wss.on('connection', (socket, req) => {
    const clientId = req.url.split('/').pop();

    if (clientId === 'host_monitor') {
        hostSocket = { socket: socket, filestream: null };
        console.log('✅ Host monitor connected');
        sendHostUpdate();

        socket.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                console.log(`[HOST COMMAND]: Received command: ${data.command}`);
                if (data.command === 'start_all') startAllRecording();
                if (data.command === 'stop_all') stopAllRecording();
            } catch (error) {
                // If JSON.parse fails, it's binary audio data
                if (hostSocket && hostSocket.filestream) {
                    hostSocket.filestream.write(message);
                } else {
                    console.warn('Received audio chunk from host, but not currently recording.');
                }
            }
        });

        socket.on('close', () => {
            console.log('❌ Host monitor disconnected');
            if (hostSocket && hostSocket.filestream) {
                hostSocket.filestream.end();
            }
            hostSocket = null;
        });

    } else { // This is a camera client
        console.log(`✅ Client ${clientId} connected`);
        clients.set(clientId, { socket, isRecording: false, filestream: null });
        sendHostUpdate();

        socket.on('message', (message) => {
            const client = clients.get(clientId);
            try {
                const data = JSON.parse(message);
                if (data.type === 'recording_fully_stopped') {
                    console.log(`[CLIENT INFO]: Client ${clientId} confirmed recording fully stopped.`);
                    client.isRecording = false;

                    if (client.filestream) {
                        client.filestream.end();
                        client.filestream = null;
                        console.log(`💾 Recording file saved for client ${clientId}`);
                    }
                    sendHostUpdate();
                }
            } catch (error) {
                // Binary video data from client
                if (client && client.isRecording && client.filestream) {
                    client.filestream.write(message);
                }
            }
        });

        socket.on('close', () => {
            console.log(`❌ Client ${clientId} disconnected`);
            const client = clients.get(clientId);
            if (client && client.filestream) {
                client.filestream.end();
            }
            clients.delete(clientId);
            sendHostUpdate();
        });
    }
});

function startAllRecording() {
    console.log('\n[SERVER ACTION]: Command received: Start all recordings');
    currentRecordingSessionPath = path.join(__dirname, 'recordings', `session_${Date.now()}`);
    fs.mkdirSync(currentRecordingSessionPath, { recursive: true });
    console.log(`[SERVER INFO]: Created new session directory: ${currentRecordingSessionPath}`);

    if (hostSocket) {
        const hostAudioFilename = `host_audio.webm`;
        const hostAudioPath = path.join(currentRecordingSessionPath, hostAudioFilename);
        hostSocket.filestream = fs.createWriteStream(hostAudioPath);
        console.log(`[HOST AUDIO]: Host audio recording started. Saving to: ${hostAudioPath}`);
    } else {
        console.warn('[SERVER WARNING]: Start recording command received, but host is not connected.');
    }

    clients.forEach((client, clientId) => {
        if (!client.isRecording) {
            const filename = `${clientId}.webm`;
            const clientVideoPath = path.join(currentRecordingSessionPath, filename);
            client.filestream = fs.createWriteStream(clientVideoPath);
            console.log(`[CLIENT VIDEO]: Client ${clientId} recording started. Saving to: ${clientVideoPath}`);
            client.isRecording = true;
            client.socket.send(JSON.stringify({ command: 'start_recording' }));
        }
    });
    sendHostUpdate();
}

function stopAllRecording() {
    console.log('\n[SERVER ACTION]: Command received: Stop all recordings');

    if (hostSocket && hostSocket.filestream) {
        hostSocket.filestream.end(() => {
            console.log('[HOST AUDIO]: Host audio stream closed.');
        });
        hostSocket.filestream = null;
    }

    clients.forEach((client, clientId) => {
        if (client.isRecording) {
            console.log(`[CLIENT VIDEO]: Sending stop command to client ${clientId}`);
            client.socket.send(JSON.stringify({ command: 'stop_recording' }));
        }
    });
    currentRecordingSessionPath = null;
}

function sendHostUpdate() {
    if (!hostSocket) return;
    const clientList = Array.from(clients.keys()).map(clientId => ({
        clientId,
        isRecording: clients.get(clientId).isRecording
    }));
    const isRecordingAll = clientList.length > 0 && clientList.every(client => client.isRecording);
    hostSocket.socket.send(JSON.stringify({
        type: 'state_update',
        isRecordingAll,
        clients: clientList
    }));
}

function getLocalIP() {
    const interfaces = os.networkInterfaces();
    for (const name of Object.keys(interfaces)) {
        for (const iface of interfaces[name]) {
            if (iface.family === 'IPv4' && !iface.internal) return iface.address;
        }
    }
    return 'localhost';
}

const PORT = 8000;
server.listen(PORT, () => {
    const ip = getLocalIP();
    console.log(`\n🚀 WebSocket server running on port ${PORT}`);
    console.log(`   Connect clients to: ws://${ip}:${PORT}`);
});