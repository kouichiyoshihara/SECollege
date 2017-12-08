var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  var message = 'Hello!'
  if (req.query.name) {
  	message = 'Hello,' + req.param('name') + '!';
  }
  res.render('index', { title: 'Express', 
  	                    message: message });
});

module.exports = router;
