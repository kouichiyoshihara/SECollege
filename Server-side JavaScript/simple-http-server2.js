var server = require('http').createServer();
var fs = require('fs');

server.on('request', (req, res) => {

	fs.readFile('./index.html', 'utf-8', (err, data) => {
		res.writeHead(200, 
		{'Content-type': 'text/html'});
		res.end(data);
	});
	
}).listen(3000);