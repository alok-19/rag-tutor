import streamlit as st

def inject_styles():
    """Inject custom premium CSS stylesheet into Streamlit."""
    st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles
       NOTE: scoped to NOT clobber Streamlit's Material Symbols icon font.
       A global `font-family` on body cascades into icon spans and replaces
       icon glyphs (e.g. the sidebar expand/collapse arrows, menu, etc.) with
       literal text like "keyboard_double_arrow_right", making those buttons
       appear broken/invisible. We explicitly restore the icon font for
       material-icon elements after setting our app font. */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    /* Restore icon glyphs (sidebar toggle, menu, app icons). Must come after
       the global font-family rule so the icon-class specificity wins. */
    [data-testid="stIconMaterial"],
    [class*="material-symbols"],
    [class*="material-icons"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stExpandSidebarButton"] span {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons", sans-serif !important;
    }
    
    /* Main container background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 10% 20%, rgba(17, 18, 30, 1) 0%, rgba(9, 10, 15, 1) 90.2%);
        color: #e2e8f0;
    }
    
    /* Header Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 16, 26, 0.95) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    
    /* Source citation cards */
    .source-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .source-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(30, 41, 59, 0.6);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
        transform: translateY(-2px);
    }
    
    /* Source badge */
    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #c084fc;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Main Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(167, 139, 250, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* ============================================================
       Premium Copilot-style Feedback Buttons
       Uses st.button() for BOTH active and inactive states (disabled=True when active).
       This guarantees identical DOM structure and zero layout shift.
       ============================================================ */
    
    /* Tighten gap in the feedback row */
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type {
        gap: 0px !important;
        align-items: center !important;
    }
    
    /* Streamlit button wrapper — kill all default spacing */
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type div[data-testid="stButton"] {
        margin: 0 !important;
        padding: 0 !important;
        width: 24px !important;
        min-width: 24px !important;
        max-width: 24px !important;
        height: 24px !important;
        min-height: 24px !important;
        max-height: 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
    }
    
    /* Ghost button base (inactive) */
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        color: rgba(148, 163, 184, 0.35) !important;
        font-family: 'Outfit', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 14px !important;
        font-weight: 300 !important;
        padding: 0 !important;
        margin: 0 !important;
        min-width: 24px !important;
        width: 24px !important;
        min-height: 24px !important;
        height: 24px !important;
        line-height: 24px !important;
        border-radius: 4px !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Streamlit inner wrappers inside the button */
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type button > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type button p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 24px !important;
    }
    
    /* Hover on inactive */
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type button:hover {
        color: rgba(148, 163, 184, 0.80) !important;
        background: transparent !important;
    }
    
    /* Pressed micro-feedback */
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type button:active {
        background: rgba(148, 163, 184, 0.06) !important;
        transform: scale(0.95) !important;
    }
    
    /* ACTIVE state — up arrow selected (disabled button in 2nd column) */
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-of-type(2) div[data-testid="stButton"] button:disabled {
        color: #34d399 !important;
        background: transparent !important;
        opacity: 1 !important;
        cursor: default !important;
    }

    /* ACTIVE state — down arrow selected (disabled button in 3rd column) */
    div[data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-of-type(3) div[data-testid="stButton"] button:disabled {
        color: #f87171 !important;
        background: transparent !important;
        opacity: 1 !important;
        cursor: default !important;
    }
    
    /* Disambiguation caption styling */
    .disambiguation-hint {
        font-style: italic;
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    /* Hide Streamlit's main hamburger menu (contains Clear cache, Rerun, etc.) */
    [data-testid="stMainMenu"] {
        display: none !important;
    }

    /* Hide Streamlit's "Deploy" button in the top-right corner.
       Targeted by testid only — never broad header-button rules, which would
       also hide the sidebar toggle. */
    [data-testid="stAppDeployButton"],
    [data-testid="stDeployButton"],
    .stAppDeployButton,
    header a[href*="share.streamlit.io"] {
        display: none !important;
    }

    /* Ensure the sidebar toggle arrows render at full visibility (their icons
       are restored by the font-family rule near the top of this stylesheet). */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* ============================================================
       New components: source pills, interactive sources, reader,
       flashcards, confidence badges, study modes
       ============================================================ */

    /* Live source pills shown during retrieval */
    .source-pills-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 6px 0 2px 0;
    }
    .src-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #c7d2fe;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 500;
        line-height: 1.4;
    }

    /* Interactive numbered source cards */
    .source-card.interactive {
        display: block;
        position: relative;
    }
    .source-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 100%);
        color: #0b0d12;
        border-radius: 50%;
        font-size: 0.72rem;
        font-weight: 700;
        margin-right: 8px;
    }
    .page-badge {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.2) !important;
        color: #a5b4fc !important;
    }
    .match-badge {
        font-size: 0.7rem !important;
    }
    .match-high { background: rgba(52, 211, 153, 0.12) !important; border-color: rgba(52, 211, 153, 0.3) !important; color: #6ee7b7 !important; }
    .match-medium { background: rgba(251, 191, 36, 0.12) !important; border-color: rgba(251, 191, 36, 0.3) !important; color: #fcd34d !important; }
    .match-low { background: rgba(248, 113, 113, 0.12) !important; border-color: rgba(248, 113, 113, 0.3) !important; color: #fca5a5 !important; }
    .source-snippet {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 8px;
        line-height: 1.45;
    }

    /* Reader pane frame */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    }

    /* Flashcard */
    .flashcard {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 28px 24px;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
        margin: 12px 0;
        transition: all 0.25s ease;
    }
    .flashcard.flipped {
        border-color: rgba(167, 139, 250, 0.5);
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.18);
    }
    .flashcard-face-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #818cf8;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .flashcard-content {
        font-size: 1.15rem;
        color: #e2e8f0;
        line-height: 1.5;
    }
    .flashcard-progress {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 16px;
    }

</style>
""", unsafe_allow_html=True)

    # JavaScript fallbacks: suppress shortcuts and hide Deploy after render.
    # NOTE: the sidebar toggle arrows are rendered correctly via the font-family
    # rule near the top of this stylesheet (restoring "Material Symbols" so the
    # arrow glyphs show instead of literal icon-name text). No JS is needed to
    # keep the toggle visible.
    st.markdown("""
<script>
    // Block Streamlit's Clear Caches shortcut (C or Shift+C) while allowing Cmd+C / Ctrl+C
    document.addEventListener('keydown', function(e) {
        if ((e.key === 'c' || e.key === 'C') && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.stopPropagation();
        }
    }, true);

    // Aggressive fallback: remove any "Deploy" button from the DOM after render
    function removeDeployButton() {
        var deployButtons = document.querySelectorAll('[data-testid="stDeployButton"], .stAppDeployButton');
        deployButtons.forEach(function(btn) { btn.style.display = 'none'; });

        var toolbar = document.querySelector('header') || document.querySelector('[data-testid="stToolbar"]');
        if (toolbar) {
            var allButtons = toolbar.querySelectorAll('button, a, div[role="button"]');
            allButtons.forEach(function(el) {
                if (el.textContent.trim() === 'Deploy') {
                    el.style.display = 'none';
                }
            });
        }
    }

    // Run immediately and after a short delay for React hydration
    removeDeployButton();
    setTimeout(removeDeployButton, 500);
    setTimeout(removeDeployButton, 1500);
</script>
""", unsafe_allow_html=True)
