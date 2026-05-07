function resizeKbChat() {
    var el = document.getElementById('kb_chatbot');
    if (!el) return;
    var fill = el.closest('.kb-chat-fill');
    if (!fill) return;
    var inputRow = fill.querySelector('.kb-input-row');
    var disc = fill.querySelector('.kb-disclaimer');
    var usedH = (inputRow ? inputRow.offsetHeight : 80) + (disc ? disc.offsetHeight : 20);
    var fillH = fill.getBoundingClientRect().height;
    el.style.height = Math.max(300, fillH - usedH - 8) + 'px';
}
setTimeout(resizeKbChat, 400);
window.addEventListener('resize', resizeKbChat);

// 阻止流式输出时自动滚到底部，保护用户阅读位置
(function() {
    var _wrap = null, _atBottom = true, _savedTop = 0;
    function _find() {
        if (_wrap) return _wrap;
        var c = document.getElementById('kb_chatbot');
        if (!c) return null;
        _wrap = c.querySelector('.bubble-wrap');
        if (_wrap) {
            _wrap.addEventListener('scroll', function() {
                var sh = _wrap.scrollHeight, st = _wrap.scrollTop, ch = _wrap.clientHeight;
                _atBottom = (sh - st - ch) < 50;
                if (!_atBottom) _savedTop = st;
            });
        }
        return _wrap;
    }
    function _keep() {
        var w = _find();
        if (!w) return;
        if (!_atBottom) { w.scrollTop = _savedTop; }
        else { w.scrollTop = w.scrollHeight; }
    }
    function _observe() {
        var w = _find();
        if (!w) { setTimeout(_observe, 300); return; }
        new MutationObserver(_keep).observe(w, {childList:true, subtree:true, characterData:true});
    }
    setTimeout(_observe, 600);
    setInterval(function() { _find(); }, 2000);
})();

// kb_query_box: Enter 发送, Shift+Enter 换行
(function() {
    function _bind() {
        var el = document.getElementById('kb_query_box');
        if (!el) { setTimeout(_bind, 300); return; }
        var ta = el.querySelector('textarea');
        if (!ta) { setTimeout(_bind, 300); return; }
        ta.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                e.stopPropagation();
                var btn = ta.closest('.kb-input-row').querySelector('button');
                if (btn) btn.click();
            } else if (e.key === 'Enter' && e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                var s = ta.selectionStart, d = ta.selectionEnd;
                ta.value = ta.value.substring(0, s) + '\n' + ta.value.substring(d);
                ta.selectionStart = ta.selectionEnd = s + 1;
                ta.dispatchEvent(new Event('input', {bubbles: true}));
            }
        }, {capture: true});
    }
    setTimeout(_bind, 500);
})();
