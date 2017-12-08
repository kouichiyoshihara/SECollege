var server = require('http').createServer();
server.on('request', (req, res) => {
	res.writeHead(200, 
		{'Content-type': 'text/plain'});
	res.end("Hello, Node.js");
}).listen(3000);