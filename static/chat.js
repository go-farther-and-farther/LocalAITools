function resizeKbChat() {
    var el = document.getElementById('kb_chatbot');
    if (!el || !document.contains(el)) return;
    var fill = el.closest('.kb-chat-fill');
    if (!fill) return;
    var inputRow = fill.querySelector('.kb-input-row');
    var disc = fill.querySelector('.kb-disclaimer');
    var usedH = (inputRow ? inputRow.offsetHeight : 80) + (disc ? disc.offsetHeight : 20);
    var fillH = fill.getBoundingClientRect().height;
    var newH = Math.max(300, fillH - usedH - 8) + 'px';
    if (el.style.height !== newH) el.style.height = newH;
}
setTimeout(resizeKbChat, 400);
window.addEventListener('resize', resizeKbChat);

// 阻止流式输出时自动滚到底部，保护用户阅读位置
(function() {
    var _wrap = null, _observer = null, _atBottom = true, _savedTop = 0;
    var _scrollBound = false, _initRetries = 0, _initTimer = null;

    function _cleanup() {
        if (_observer) { _observer.disconnect(); _observer = null; }
        _wrap = null;
        _scrollBound = false;
    }

    function _ensureScrollListener() {
        if (!_wrap || _scrollBound) return;
        _wrap.addEventListener('scroll', function() {
            var sh = _wrap.scrollHeight, st = _wrap.scrollTop, ch = _wrap.clientHeight;
            _atBottom = (sh - st - ch) < 50;
            if (!_atBottom) _savedTop = st;
        });
        _scrollBound = true;
    }

    function _ensureObserver() {
        if (_observer || !_wrap) return;
        _observer = new MutationObserver(function() {
            if (!_wrap || !document.contains(_wrap)) {
                _cleanup();
                return;
            }
            if (!_atBottom) { _wrap.scrollTop = _savedTop; }
            else { _wrap.scrollTop = _wrap.scrollHeight; }
        });
        _observer.observe(_wrap, {childList:true, subtree:true, characterData:true});
    }

    function _find() {
        // Return cached element if still valid
        if (_wrap && document.contains(_wrap)) return _wrap;
        // Element removed from DOM
        if (_wrap) _cleanup();
        var c = document.getElementById('kb_chatbot');
        if (!c) return null;
        _wrap = c.querySelector('.bubble-wrap');
        if (_wrap) {
            _ensureScrollListener();
            _ensureObserver();
        }
        return _wrap;
    }

    // Detect on init; stop polling once found (Gradio tabs hide but don't remove DOM)
    function _tryInit() {
        _find();
        if (_wrap) { _initRetries = 0; return; }
        _initRetries++;
        if (_initRetries > 50) return;  // give up after ~30s
        _initTimer = setTimeout(_tryInit, 600);
    }
    _tryInit();
})();

// ===== Theme color =====
(function() {
    function hexToHsl(hex) {
        hex = hex.replace('#', '');
        var r = parseInt(hex.substring(0,2),16)/255;
        var g = parseInt(hex.substring(2,4),16)/255;
        var b = parseInt(hex.substring(4,6),16)/255;
        var max = Math.max(r,g,b), min = Math.min(r,g,b), h, s, l = (max+min)/2;
        if (max === min) { h = s = 0; }
        else {
            var d = max - min;
            s = l > 0.5 ? d/(2-max-min) : d/(max+min);
            switch (max) {
                case r: h = ((g-b)/d + (g<b?6:0))/6; break;
                case g: h = ((b-r)/d + 2)/6; break;
                case b: h = ((r-g)/d + 4)/6; break;
            }
        }
        return [Math.round(h*360), Math.round(s*100), Math.round(l*100)];
    }
    function hslStr(h,s,l) { return 'hsl('+h+','+s+'%,'+l+'%)'; }

    function applyTheme(hex) {
        var hsl = hexToHsl(hex);
        var h = hsl[0], s = hsl[1], l = hsl[2];
        var root = document.documentElement.style;
        root.setProperty('--accent', hex);
        root.setProperty('--accent-light', hslStr(h, Math.min(s,60), Math.min(95, l+40)));
        root.setProperty('--accent-grad-a', hslStr(h, Math.min(s+10,90), Math.max(45, l)));
        root.setProperty('--accent-grad-b', hslStr((h+40)%360, Math.min(s,80), Math.max(35, l-10)));
    }

    // Apply saved theme on load
    var saved = localStorage.getItem('theme_accent');
    if (saved) applyTheme(saved);

    // Expose for Gradio
    window.setThemeColor = function(hex) {
        localStorage.setItem('theme_accent', hex);
        applyTheme(hex);
    };
})();

// kb_query_box: Enter 发送, Shift+Enter 换行
(function() {
    var _bound = false;
    function _bind() {
        if (_bound) return;
        var el = document.getElementById('kb_query_box');
        if (!el) { setTimeout(_bind, 500); return; }
        var ta = el.querySelector('textarea');
        if (!ta) { setTimeout(_bind, 500); return; }
        _bound = true;
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
