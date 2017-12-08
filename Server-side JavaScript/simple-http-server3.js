var server = require('http').createServer();
var fs = require('fs');
var url = require('url');

server.on('request', (req, res) => {

	var pathname = url.parse(req.url).pathname;
	if (pathname === "/") {
		pathname = "/index.html";
	}

	fs.readFile("." + pathname, 'utf-8', (err, data) => {
		res.writeHead(200, 
		{'Content-type': 'text/html'});
		res.end(data);
	});
	
}).listen(3000);