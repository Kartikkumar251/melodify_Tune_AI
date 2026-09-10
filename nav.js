/* nav.js — Melodyfy shared navigation (single source of truth) */
(function () {

  /* ── 1. Inject styles once ── */
  if (!document.getElementById('mfy-nav-styles')) {
    var s = document.createElement('style');
    s.id = 'mfy-nav-styles';
    s.textContent = [
      /* keyframes */
      '@keyframes nav-slide-in{from{transform:translateY(-100%);opacity:0}to{transform:translateY(0);opacity:1}}',
      '@keyframes link-pop{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}',
      '@keyframes mob-open{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}',

      /* nav shell */
      'nav.mfy-nav{',
        'position:sticky;top:0;z-index:200;',
        'height:56px;',
        'display:flex;align-items:center;padding:0 24px;gap:0;',
        'background:rgba(7,7,10,.82);',
        'backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);',
        'border-bottom:1px solid rgba(237,232,223,.08);',
        'box-shadow:0 1px 0 rgba(237,232,223,.04),0 4px 24px rgba(0,0,0,.35);',
        'animation:nav-slide-in .35s cubic-bezier(.22,1,.36,1) both;',
      '}',

      /* logo */
      '.mfy-logo{display:flex;align-items:center;gap:9px;text-decoration:none;',
        'font-family:"Archivo Black",sans-serif;font-size:16px;letter-spacing:-.01em;',
        'color:var(--text-primary);flex-shrink:0;margin-right:28px;',
        'transition:opacity .2s;}',
      '.mfy-logo:hover{opacity:.8;}',
      '.mfy-logo-icon{width:28px;height:28px;flex-shrink:0;}',

      /* links */
      '.mfy-links{display:flex;align-items:center;gap:2px;flex:1;}',
      '.mfy-links a{',
        'position:relative;color:var(--text-muted);text-decoration:none;',
        'padding:6px 11px;border-radius:6px;font-size:13.5px;font-weight:500;',
        'letter-spacing:-.01em;white-space:nowrap;',
        'transition:color .18s,background .18s;',
        'animation:link-pop .3s cubic-bezier(.22,1,.36,1) both;',
      '}',
      '.mfy-links a:nth-child(1){animation-delay:.04s}',
      '.mfy-links a:nth-child(2){animation-delay:.08s}',
      '.mfy-links a:nth-child(3){animation-delay:.12s}',
      '.mfy-links a:nth-child(4){animation-delay:.16s}',
      '.mfy-links a:nth-child(5){animation-delay:.20s}',
      '.mfy-links a:nth-child(6){animation-delay:.24s}',
      '.mfy-links a:nth-child(7){animation-delay:.28s}',
      '.mfy-links a::after{',
        'content:"";position:absolute;left:50%;bottom:3px;',
        'width:0;height:2px;border-radius:2px;',
        'background:var(--accent);',
        'transform:translateX(-50%);',
        'transition:width .22s cubic-bezier(.22,1,.36,1);',
      '}',
      '.mfy-links a:hover{color:var(--text-primary);background:rgba(237,232,223,.06);}',
      '.mfy-links a.active{color:var(--text-primary);}',
      '.mfy-links a.active::after{width:calc(100% - 22px);}',

      /* spacer */
      '.mfy-spacer{flex:1;}',

      /* right actions */
      '.mfy-actions{display:flex;align-items:center;gap:8px;flex-shrink:0;}',

      /* base btn (shared across pages) */
      '.btn{display:inline-flex;align-items:center;gap:6px;padding:5px 16px;',
        'border-radius:6px;font-size:13.5px;font-weight:500;cursor:pointer;',
        'border:1px solid transparent;transition:all .2s cubic-bezier(.22,1,.36,1);',
        'text-decoration:none;font-family:"Space Grotesk",sans-serif;',
        'letter-spacing:-.01em;white-space:nowrap;}',
      '.btn:disabled{opacity:.45;cursor:not-allowed;}',

      /* ghost */
      '.btn-ghost{background:transparent;border-color:rgba(237,232,223,.12);color:var(--text-secondary);}',
      '.btn-ghost:hover{background:rgba(237,232,223,.07);border-color:rgba(237,232,223,.22);color:var(--text-primary);}',

      /* accent (New Beat) */
      '.btn-green{background:var(--accent);border-color:transparent;color:#07070a;font-weight:600;',
        'box-shadow:0 0 0 0 rgba(237,232,223,0);}',
      '.btn-green:hover{background:var(--accent-hover);',
        'transform:translateY(-1px);',
        'box-shadow:0 4px 18px rgba(237,232,223,.22);}',
      '.btn-green:active{transform:translateY(0);box-shadow:none;}',

      /* sign-out — text-only link style */
      '.btn-signout{background:transparent;border-color:transparent;',
        'color:var(--text-muted);font-size:13px;padding:5px 10px;}',
      '.btn-signout:hover{color:var(--red,#f87171);background:rgba(248,113,113,.06);',
        'border-color:transparent;}',

      /* settings icon btn */
      '.btn-icon{width:32px;height:32px;padding:0;border-radius:7px;',
        'justify-content:center;',
        'background:transparent;border-color:rgba(237,232,223,.1);color:var(--text-muted);}',
      '.btn-icon:hover{background:rgba(237,232,223,.07);color:var(--text-primary);',
        'border-color:rgba(237,232,223,.2);}',
      '.btn-icon svg{width:15px;height:15px;flex-shrink:0;}',

      /* size variants */
      '.btn-sm{padding:3px 11px;font-size:12.5px;}',

      /* coloured variants used in studio / explore */
      '.btn-blue{background:rgba(88,166,255,.12);border-color:rgba(88,166,255,.25);color:#7eb8f7;}',
      '.btn-blue:hover{background:rgba(88,166,255,.22);}',
      '.btn-orange{background:rgba(240,136,62,.12);border-color:rgba(240,136,62,.25);color:var(--orange,#f0883e);}',
      '.btn-orange:hover{background:rgba(240,136,62,.22);}',
      '.btn-purple{background:rgba(188,140,255,.12);border-color:rgba(188,140,255,.25);color:var(--purple,#bc8cff);}',
      '.btn-purple:hover{background:rgba(188,140,255,.22);}',
      '.btn-danger{background:rgba(248,113,113,.1);border-color:rgba(248,113,113,.28);color:var(--red,#f87171);}',
      '.btn-danger:hover{background:rgba(248,113,113,.2);}',

      /* divider between links and actions */
      '.mfy-divider{width:1px;height:20px;background:rgba(237,232,223,.1);margin:0 4px;flex-shrink:0;}',

      /* mobile hamburger */
      '.mfy-burger{display:none;flex-direction:column;justify-content:center;',
        'gap:5px;width:36px;height:36px;padding:4px;border-radius:6px;',
        'background:transparent;border:1px solid rgba(237,232,223,.1);',
        'cursor:pointer;margin-left:auto;}',
      '.mfy-burger span{display:block;height:1.5px;border-radius:2px;',
        'background:var(--text-secondary);transition:all .25s cubic-bezier(.22,1,.36,1);}',
      '.mfy-burger.open span:nth-child(1){transform:translateY(6.5px) rotate(45deg);}',
      '.mfy-burger.open span:nth-child(2){opacity:0;transform:scaleX(0);}',
      '.mfy-burger.open span:nth-child(3){transform:translateY(-6.5px) rotate(-45deg);}',

      /* mobile drawer */
      '.mfy-mob{display:none;flex-direction:column;gap:2px;',
        'padding:10px 16px 16px;',
        'background:rgba(7,7,10,.96);backdrop-filter:blur(24px);',
        'border-bottom:1px solid rgba(237,232,223,.08);animation:mob-open .25s ease both;}',
      '.mfy-mob.open{display:flex;}',
      '.mfy-mob a,.mfy-mob button{',
        'padding:10px 12px;border-radius:7px;font-size:14px;',
        'color:var(--text-secondary);text-decoration:none;',
        'border:none;background:transparent;',
        'cursor:pointer;text-align:left;font-family:"Space Grotesk",sans-serif;',
        'transition:background .15s,color .15s;}',
      '.mfy-mob a:hover,.mfy-mob button:hover{background:rgba(237,232,223,.07);color:var(--text-primary);}',
      '.mfy-mob a.active{color:var(--text-primary);background:rgba(237,232,223,.09);}',
      '.mfy-mob .mob-divider{height:1px;background:rgba(237,232,223,.07);margin:6px 0;}',
      '.mfy-mob .mob-new-beat{background:var(--accent);color:#07070a;',
        'font-weight:600;margin-top:4px;}',
      '.mfy-mob .mob-new-beat:hover{background:var(--accent-hover);}',

      /* scroll-shrink: slightly compress nav on scroll */
      'nav.mfy-nav.scrolled{height:48px;background:rgba(7,7,10,.92);}',

      /* responsive */
      '@media(max-width:768px){',
        '.mfy-links,.mfy-actions,.mfy-divider{display:none;}',
        '.mfy-burger{display:flex;}',
      '}',
    ].join('');
    (document.head || document.documentElement).appendChild(s);
  }

  /* ── 2. Build nav ── */
  var placeholder = document.getElementById('nav-root');
  if (!placeholder) return;

  var page = (location.pathname.split('/').pop() || 'index.html').replace(/[?#].*$/, '');
  var isLoggedIn = !!localStorage.getItem('bf_token');

  var ALL_LINKS = [
    { href: 'dashboard.html',  label: 'Dashboard',  auth: true  },
    { href: 'explore.html',    label: 'Explore',     auth: false },
    { href: 'studio.html',     label: 'Studio',      auth: false },
    { href: 'community.html',  label: 'Community',   auth: true  },
    { href: 'library.html',    label: 'Library',     auth: true  },
    { href: 'projects.html',   label: 'Projects',    auth: true  },
    { href: 'repo.html',       label: 'Repository',  auth: true  },
  ];

  var visibleLinks = ALL_LINKS.filter(function(l){ return !l.auth || isLoggedIn; });

  var desktopLinks = visibleLinks.map(function(l){
    var cls = l.href === page ? ' class="active"' : '';
    return '<a href="' + l.href + '"' + cls + '>' + l.label + '</a>';
  }).join('');

  var mobileLinks = visibleLinks.map(function(l){
    var cls = l.href === page ? ' class="active"' : '';
    return '<a href="' + l.href + '"' + cls + '>' + l.label + '</a>';
  }).join('');

  /* SVG icons */
  var iconSettings = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="2.5"/><path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3.05 3.05l1.06 1.06M11.89 11.89l1.06 1.06M3.05 12.95l1.06-1.06M11.89 4.11l1.06-1.06"/></svg>';

  /* New Beat icon — a simple + waveform */
  var newBeatLabel = '+ New Beat';

  /* right side */
  var desktopRight, mobileExtra;
  if (isLoggedIn) {
    desktopRight =
      '<div class="mfy-divider"></div>' +
      '<a href="studio.html" class="btn btn-green btn-sm">' + newBeatLabel + '</a>' +
      '<a href="settings.html" class="btn btn-icon btn-sm" title="Settings" aria-label="Settings">' + iconSettings + '</a>' +
      '<button class="btn btn-signout" onclick="(function(){localStorage.removeItem(\'bf_token\');location.href=\'index.html\';})()">Sign out</button>';
    mobileExtra =
      '<div class="mob-divider"></div>' +
      '<a href="studio.html" class="mob-new-beat">' + newBeatLabel + '</a>' +
      '<a href="settings.html">Settings</a>' +
      '<button onclick="(function(){localStorage.removeItem(\'bf_token\');location.href=\'index.html\';})()">Sign out</button>';
  } else {
    desktopRight =
      '<div class="mfy-divider"></div>' +
      '<button class="btn btn-ghost btn-sm" onclick="(typeof openModal===\'function\'&&document.getElementById(\'signin-modal\'))?openModal(\'signin-modal\'):location.href=\'index.html?auth=signin\'">Sign in</button>' +
      '<button class="btn btn-green btn-sm" onclick="(typeof openModal===\'function\'&&document.getElementById(\'register-modal\'))?openModal(\'register-modal\'):location.href=\'index.html?auth=register\'">Get started</button>';
    mobileExtra =
      '<div class="mob-divider"></div>' +
      '<button onclick="(typeof openModal===\'function\'&&document.getElementById(\'signin-modal\'))?openModal(\'signin-modal\'):location.href=\'index.html?auth=signin\'">Sign in</button>' +
      '<button onclick="(typeof openModal===\'function\'&&document.getElementById(\'register-modal\'))?openModal(\'register-modal\'):location.href=\'index.html?auth=register\'" class="mob-new-beat">Get started</button>';
  }

  /* logo SVG */
  var logoSVG =
    '<svg class="mfy-logo-icon" viewBox="0 0 28 28" fill="none">' +
    '<circle cx="14" cy="14" r="13" stroke="var(--accent)" stroke-width="1.8"/>' +
    '<path d="M8 18 Q11 10 14 14 Q17 18 20 11" stroke="var(--accent)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>' +
    '</svg>';

  /* Build wrapper: nav + mobile drawer */
  var wrapper = document.createElement('div');
  wrapper.id = 'mfy-nav-wrapper';

  var nav = document.createElement('nav');
  nav.className = 'mfy-nav';
  nav.innerHTML =
    '<a class="mfy-logo" href="index.html">' + logoSVG + 'Melodyfy</a>' +
    '<div class="mfy-links">' + desktopLinks + '</div>' +
    '<div class="mfy-spacer"></div>' +
    '<div class="mfy-actions">' + desktopRight + '</div>' +
    '<button class="mfy-burger" id="mfy-burger" aria-label="Menu" aria-expanded="false">' +
      '<span></span><span></span><span></span>' +
    '</button>';

  var mob = document.createElement('div');
  mob.className = 'mfy-mob';
  mob.id = 'mfy-mob';
  mob.innerHTML = mobileLinks + mobileExtra;

  wrapper.appendChild(nav);
  wrapper.appendChild(mob);
  placeholder.replaceWith(wrapper);

  /* ── 3. Burger toggle ── */
  var burger = document.getElementById('mfy-burger');
  var mobMenu = document.getElementById('mfy-mob');
  if (burger && mobMenu) {
    burger.addEventListener('click', function() {
      var open = mobMenu.classList.toggle('open');
      burger.classList.toggle('open', open);
      burger.setAttribute('aria-expanded', String(open));
    });
    /* close on outside click */
    document.addEventListener('click', function(e) {
      if (!wrapper.contains(e.target)) {
        mobMenu.classList.remove('open');
        burger.classList.remove('open');
        burger.setAttribute('aria-expanded', 'false');
      }
    });
  }

  /* ── 4. Scroll-shrink ── */
  var navEl = nav;
  var ticking = false;
  window.addEventListener('scroll', function() {
    if (!ticking) {
      requestAnimationFrame(function() {
        navEl.classList.toggle('scrolled', window.scrollY > 20);
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });

})();
