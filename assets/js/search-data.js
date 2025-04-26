// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "publications by categories in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-an-intro-to-uv",
      
        title: "An intro to uv",
      
      description: "A Swiss Army Knife for Python data science",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-uv/";
        
      },
    },{id: "post-shap-values",
      
        title: "SHAP values",
      
      description: "A model-agnostic framework for explaining predictions",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/shapley/";
        
      },
    },{id: "post-knockoffs",
      
        title: "Knockoffs",
      
      description: "FDR-controlled feature selection",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/kernel-knockoffs/";
        
      },
    },{id: "post-random-walks-and-markov-chains",
      
        title: "Random walks and Markov chains",
      
      description: "PageRank, MCMC, and others",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/graphs-random-walks/";
        
      },
    },{id: "post-graphs-and-linear-algebra",
      
        title: "Graphs and Linear Algebra",
      
      description: "Matrices associated to graphs and their properties",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/graphs-linear-algebra/";
        
      },
    },{id: "post-properties-of-graphs",
      
        title: "Properties of Graphs",
      
      description: "Multiscale ways to talk about graphs",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/graph-properties/";
        
      },
    },{id: "post-graph-glossary",
      
        title: "Graph Glossary",
      
      description: "Definitions of frequent graph terms",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/graphs-glossary/";
        
      },
    },{id: "post-introduction-to-graphs",
      
        title: "Introduction to Graphs",
      
      description: "Basic definitions",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/graphs-basics/";
        
      },
    },{id: "post-python-functions",
      
        title: "Python functions",
      
      description: "Fourth post in the Python series",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-functions/";
        
      },
    },{id: "post-data-structures-and-algorithms",
      
        title: "Data Structures and Algorithms",
      
      description: "Common Problems and How to Solve Them",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/data-structures-algorithms/";
        
      },
    },{id: "post-pandas",
      
        title: "Pandas",
      
      description: "Cute bear, ok library",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-pandas/";
        
      },
    },{id: "post-quirks-of-python",
      
        title: "Quirks of Python",
      
      description: "A catch-all of interesting behaviors",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-quirks/";
        
      },
    },{id: "post-how-computers-work",
      
        title: "How computers work",
      
      description: "An introduction of CPUs and RAM for data scientists",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/hardware/";
        
      },
    },{id: "post-python-dictionaries",
      
        title: "Python dictionaries",
      
      description: "Keeping Python together since 1991",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-dicts/";
        
      },
    },{id: "post-python-lists-and-tuples",
      
        title: "Python lists and tuples",
      
      description: "Buckets, buckets, buckets!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-lists/";
        
      },
    },{id: "post-vectors-and-matrices-in-python",
      
        title: "Vectors and matrices in Python",
      
      description: "NumPy and vectorization",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-vectors/";
        
      },
    },{id: "post-the-basics-of-python",
      
        title: "The Basics of Python",
      
      description: "Revisiting Python&#39;s properties",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-basics/";
        
      },
    },{id: "post-python-objects",
      
        title: "Python objects",
      
      description: "Everything is an object!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/python-objects/";
        
      },
    },{id: "post-data-centric-machine-learning",
      
        title: "Data-centric machine learning",
      
      description: "Notes from the MIT course",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/data-centric-ml/";
        
      },
    },{id: "post-mendelian-randomization",
      
        title: "Mendelian randomization",
      
      description: "Causality.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/mendelian-randomization/";
        
      },
    },{id: "post-finding-similar-documents",
      
        title: "Finding similar documents",
      
      description: "A fast algorithm to quickly scan over the corpus",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/finding-similar-items/";
        
      },
    },{id: "post-familial-breast-cancer",
      
        title: "Familial breast cancer",
      
      description: "Molecular basis of familial breast cancer",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/familial-breast-cancer/";
        
      },
    },{id: "post-heritability",
      
        title: "Heritability",
      
      description: "A first approach to the heritability of complex traits",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/heritability/";
        
      },
    },{id: "news-i-joined-the-novo-nordisk-research-center-oxford-as-senior-data-scientist-i-will-be-applying-machine-learning-to-genetics",
          title: 'I joined the Novo Nordisk Research Center Oxford as Senior Data Scientist. I...',
          description: "",
          section: "News",},{id: "news-our-preprint-on-detecting-epistasis-using-sparkles-quantum-computing-sparkles-was-just-published-on-medrxiv",
          title: 'Our preprint on detecting epistasis using :sparkles: quantum computing :sparkles: was just published...',
          description: "",
          section: "News",},{id: "news-our-preprint-on-predicting-cardiovascular-disease-risk-using-interpretable-ml-is-out-this-work-was-conducted-in-partnership-with-microsoft-research",
          title: 'Our preprint on predicting cardiovascular disease risk using interpretable ML is out! This...',
          description: "",
          section: "News",},{id: "news-our-model-for-cardiovascular-disease-risk-prediction-got-highlighted-in-novo-nordisk-s-capital-markets-day",
          title: 'Our model for cardiovascular disease risk prediction got highlighted in Novo Nordisk’s Capital...',
          description: "",
          section: "News",},{id: "news-i-was-promoted-to-lead-data-scientist-i-will-be-working-on-ai-ml-for-target-and-biomarker-discovery",
          title: 'I was promoted to Lead Data Scientist. I will be working on AI/ML...',
          description: "",
          section: "News",},{
        id: 'social-bluesky',
        title: 'Bluesky',
        section: 'Socials',
        handler: () => {
          window.open("https://bsky.app/profile/hclimente.eu", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%68%69@%68%63%6C%69%6D%65%6E%74%65.%65%75", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/hclimente", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/hclimente", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0000-0002-3030-7471", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=tK7i7zwAAAAJ", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/hclimente", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
