self.addEventListener('message', function(event) {

	// 何らかの時間のかかる処理のシミュレーション
	for(var i = 0; i < 100000; i++) {
		for(var j = 0; j < 50000; j++) {
			
		}
	}

	self.postMessage("The result is " + event.data * 2);

}, false);


