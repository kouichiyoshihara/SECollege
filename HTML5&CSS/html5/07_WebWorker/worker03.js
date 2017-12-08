self.addEventListener('message', function(event) {
	
	var value = 0;
	
	//  0.5秒間隔で繰り返し
	var timer = setInterval(function() {
	
		// 100までの値をランダムで進捗率にプラスする
		value += Math.random() * 100;
		self.postMessage(value);

		if(progress.value >= progress.max) {
			clearInterval(timer); // 終了
			return;
		}
		
	}, 500);
			
}, false);