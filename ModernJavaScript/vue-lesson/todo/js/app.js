(function(exports) {

	'use strict';

	exports.app = new Vue({
		el: '.todoapp',
		data: {
			todos: todoStorage.fetch(),
			newTodo: ''
		},
		methods: {
			addTodo: function() {
				var value = this.newTodo.trim();
				if (!value) { return; }
				this.todos.push({title: value});
				this.newTodo = '';
			}
		},
		watch: {
			todos: {
				deep: true,
				handler: todoStorage.save
			}
		}
	});

})(window);
