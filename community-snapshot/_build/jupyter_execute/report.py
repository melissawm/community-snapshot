#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path as op
import seaborn as sns
import pandas as pd
import numpy as np
import altair as alt
import nbformat as nbf
import github_activity as ga
from datetime import timedelta, date
from pathlib import Path
from dateutil.relativedelta import relativedelta
from myst_nb import glue
from markdown import markdown
from IPython.display import Markdown
from ipywidgets.widgets import HTML, Tab
from ipywidgets import widgets
from matplotlib import pyplot as plt

from warnings import simplefilter
simplefilter('ignore')


# In[2]:


# Parameters to use later
github_orgs = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
]


# # Repository information

# In[3]:


# Last 3 months
n_days = 90


# ## Fetch latest data
# 
# In this case, we do not save the output notebook because it's just meant for execution.

# In[4]:


get_ipython().run_line_magic('env', 'GITHUB_ACCESS_TOKEN=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')


# In[5]:


github_org = "numpy"
top_n_repos = 15
n_days = 10


# In[6]:


stop = date.today()
start = date.today() - relativedelta(days=n_days)

# Strings for use in queries
start_date = f"{start:%Y-%m-%d}"
stop_date = f"{stop:%Y-%m-%d}"


# In[7]:


raw_data = ga.get_activity(github_org, start_date)
data = raw_data.copy()

# Prepare our data
data["kind"] = data["url"].map(lambda a: "issue" if "issues/" in a else "pr")
data["mergedBy"] = data["mergedBy"].map(lambda a: a["login"] if not isinstance(a, (float, type(None))) else None)

prs = data.query("kind == 'pr'")
issues = data.query("kind == 'issue'")

# Pull out the comments
comments = []
for _, irow in data.iterrows():
    for icomment in irow['comments']['edges']:
        icomment = icomment["node"].copy()
        icomment["author"] = icomment["author"]["login"] if icomment["author"] else None
        icomment["org"] = irow["org"]
        icomment["repo"] = irow["repo"]
        icomment["id"] = irow["id"]
        comments.append(pd.Series(icomment))
comments = pd.DataFrame(comments)

# Clean up
for idata in [prs, comments, issues]:
    idata.drop_duplicates(subset=["url"], inplace=True)

# What are the top N repos, we will only plot these in the full data plots
top_commented_repos = comments.groupby("repo").count().sort_values("createdAt", ascending=False)['createdAt']
use_repos = top_commented_repos.head(top_n_repos).index.tolist()


# In[8]:


from pathlib import Path
path_data = Path("../data/")
path_data.mkdir(exist_ok=True)
for kind, idata in [("comments", comments), ("prs", prs), ("issues", issues)]:
    path_data = Path(f"../data/{kind}.csv")
    if path_data.exists():
        idata = pd.read_csv(path_data).append(idata)
    idata = idata.drop_duplicates(subset=["url"])
    idata.to_csv(path_data, index=None)


# ## Generate a book from the data

# In[9]:


# Altair config
def author_url(author):
    return f"https://github.com/{author}"

def alt_theme():
    return {
        'config': {
            'axisLeft': {
                'labelFontSize': 15,
            },
            'axisBottom': {
                'labelFontSize': 15,
            },
        }
    }



# Define colors we'll use for GitHub membership
author_types = ['MEMBER', 'CONTRIBUTOR', 'COLLABORATOR', "NONE"]

author_palette = np.array(sns.palettes.blend_palette(["lightgrey", "lightgreen", "darkgreen"], 4)) * 256
author_colors = ["rgb({}, {}, {})".format(*color) for color in author_palette]
author_color_dict = {key: val for key, val in zip(author_types, author_palette)}


# In[10]:


# Glue variables for use in markdown
glue(f"{github_org}_github_org", github_org, display=False)
glue(f"{github_org}_start", start_date, display=False)
glue(f"{github_org}_stop", stop_date, display=False)


# In[11]:


from pathlib import Path
path_data = Path("../data")
comments = pd.read_csv(path_data.joinpath('comments.csv'), index_col=None).drop_duplicates()
issues = pd.read_csv(path_data.joinpath('issues.csv'), index_col=None).drop_duplicates()
prs = pd.read_csv(path_data.joinpath('prs.csv'), index_col=None).drop_duplicates()

for idata in [comments, issues, prs]:
    idata.query("org == @github_org", inplace=True)


# In[12]:


# What are the top N repos, we will only plot these in the full data plots
top_commented_repos = comments.groupby("repo").count().sort_values("createdAt", ascending=False)['createdAt']
use_repos = top_commented_repos.head(top_n_repos).index.tolist()


# In[13]:


merged = prs.query('state == "MERGED" and closedAt > @start_date and closedAt < @stop_date')


# In[14]:


prs_by_repo = merged.groupby(['org', 'repo']).count()['author'].reset_index().sort_values(['org', 'author'], ascending=False)
alt.Chart(data=prs_by_repo, title=f"Merged PRs in the last {n_days} days").mark_bar().encode(
    x=alt.X('repo', sort=prs_by_repo['repo'].values.tolist()),
    y='author',
    color='org'
)


# ### Authoring and merging stats by repository
# 
# Let's see who has been doing most of the PR authoring and merging. The PR author is generally the
# person that implemented a change in the repository (code, documentation, etc). The PR merger is
# the person that "pressed the green button" and got the change into the main codebase.

# In[15]:


# Prep our merging DF
merged_by_repo = merged.groupby(['repo', 'author'], as_index=False).agg({'id': 'count', 'authorAssociation': 'first'}).rename(columns={'id': "authored", 'author': 'username'})
closed_by_repo = merged.groupby(['repo', 'mergedBy']).count()['id'].reset_index().rename(columns={'id': "closed", "mergedBy": "username"})


# In[16]:


charts = []
title = f"PR authors for {github_org} in the last {n_days} days"
this_data = merged_by_repo.replace(np.nan, 0).groupby('username', as_index=False).agg({'authored': 'sum', 'authorAssociation': 'first'})
this_data = this_data.sort_values('authored', ascending=False)
ch = alt.Chart(data=this_data, title=title).mark_bar().encode(
    x='username',
    y='authored',
    color=alt.Color('authorAssociation', scale=alt.Scale(domain=author_types, range=author_colors))
)
ch


# In[17]:


charts = []
title = f"Merges for {github_org} in the last {n_days} days"
ch = alt.Chart(data=closed_by_repo.replace(np.nan, 0), title=title).mark_bar().encode(
    x='username',
    y='closed',
)
ch


# ### Issues

# In[18]:


created = issues.query('state == "OPEN" and createdAt > @start_date and createdAt < @stop_date')
closed = issues.query('state == "CLOSED" and closedAt > @start_date and closedAt < @stop_date')


# In[19]:


created_counts = created.groupby(['org', 'repo']).count()['number'].reset_index()
created_counts['org/repo'] = created_counts.apply(lambda a: a['org'] + '/' + a['repo'], axis=1)
sorted_vals = created_counts.sort_values(['org', 'number'], ascending=False)['repo'].values
alt.Chart(data=created_counts, title=f"Issues created in the last {n_days} days").mark_bar().encode(
    x=alt.X('repo', sort=alt.Sort(sorted_vals.tolist())),
    y='number',
)


# In[20]:


closed_counts = closed.groupby(['org', 'repo']).count()['number'].reset_index()
closed_counts['org/repo'] = closed_counts.apply(lambda a: a['org'] + '/' + a['repo'], axis=1)
sorted_vals = closed_counts.sort_values(['number'], ascending=False)['repo'].values
alt.Chart(data=closed_counts, title=f"Issues closed in the last {n_days} days").mark_bar().encode(
    x=alt.X('repo', sort=alt.Sort(sorted_vals.tolist())),
    y='number',
)


# In[21]:


created_closed = pd.merge(created_counts.rename(columns={'number': 'created'}).drop(columns='org/repo'),
                          closed_counts.rename(columns={'number': 'closed'}).drop(columns='org/repo'),
                          on=['org', 'repo'], how='outer')

created_closed = pd.melt(created_closed, id_vars=['org', 'repo'], var_name="kind", value_name="count").replace(np.nan, 0)


# In[22]:


charts = []
# Pick the top 10 repositories
top_repos = created_closed.groupby(['repo']).sum().sort_values(by='count', ascending=False).head(10).index
ch = alt.Chart(created_closed.query('repo in @top_repos'), width=120).mark_bar().encode(
    x=alt.X("kind", axis=alt.Axis(labelFontSize=15, title="")), 
    y=alt.Y('count', axis=alt.Axis(titleFontSize=15, labelFontSize=12)),
    color='kind',
    column=alt.Column("repo", header=alt.Header(title=f"Issue activity, last {n_days} days for {github_org}", titleFontSize=15, labelFontSize=12))
)
ch


# In[23]:


# Set to datetime
for kind in ['createdAt', 'closedAt']:
    closed.loc[:, kind] = pd.to_datetime(closed[kind])
    
closed.loc[:, 'time_open'] = closed['closedAt'] - closed['createdAt']
closed.loc[:, 'time_open'] = closed['time_open'].dt.total_seconds()


# In[24]:


time_open = closed.groupby(['org', 'repo']).agg({'time_open': 'median'}).reset_index()
time_open['time_open'] = time_open['time_open'] / (60 * 60 * 24)
time_open['org/repo'] = time_open.apply(lambda a: a['org'] + '/' + a['repo'], axis=1)
sorted_vals = time_open.sort_values(['org', 'time_open'], ascending=False)['repo'].values
alt.Chart(data=time_open, title=f"Time to close for issues closed in the last {n_days} days").mark_bar().encode(
    x=alt.X('repo', sort=alt.Sort(sorted_vals.tolist())),
    y=alt.Y('time_open', title="Median Days Open"),
)


# ## Most-upvoted issues

# In[25]:


thumbsup = issues.sort_values("thumbsup", ascending=False).head(25)
thumbsup = thumbsup[["title", "url", "number", "thumbsup", "repo"]]

text = []
for ii, irow in thumbsup.iterrows():
    itext = f"- ({irow['thumbsup']}) {irow['title']} - {irow['repo']} - [#{irow['number']}]({irow['url']})"
    text.append(itext)
text = '\n'.join(text)
HTML(markdown(text))


# ## Commenters across repositories
# 
# These are commenters across all issues and pull requests in the last several days.
# These are colored by the commenter's association with the organization. For information
# about what these associations mean, [see this StackOverflow post](https://stackoverflow.com/a/28866914/1927102).

# In[26]:


commentors = (
    comments
    .query("createdAt > @start_date and createdAt < @stop_date")
    .groupby(['org', 'repo', 'author', 'authorAssociation'])
    .count().rename(columns={'id': 'count'})['count']
    .reset_index()
    .sort_values(['org', 'count'], ascending=False)
)


# In[27]:


n_plot = 50
charts = []
for ii, (iorg, idata) in enumerate(commentors.groupby(['org'])):
    title = f"Top {n_plot} commentors for {iorg} in the last {n_days} days"
    idata = idata.groupby('author', as_index=False).agg({'count': 'sum', 'authorAssociation': 'first'})
    idata = idata.sort_values('count', ascending=False).head(n_plot)
    ch = alt.Chart(data=idata.head(n_plot), title=title).mark_bar().encode(
        x='author',
        y='count',
        color=alt.Color('authorAssociation', scale=alt.Scale(domain=author_types, range=author_colors))
    )
    charts.append(ch)
alt.hconcat(*charts)


# ## First responders
# 
# First responders are the first people to respond to a new issue in one of the repositories.
# The following plots show first responders for recently-created issues.

# In[28]:


first_comments = []
for (org, repo, issue_id), i_comments in comments.groupby(['org', 'repo', 'id']):
    ix_min = pd.to_datetime(i_comments['createdAt']).idxmin()
    first_comment = i_comments.loc[ix_min]
    if isinstance(first_comment, pd.DataFrame):
        first_comment = first_comment.iloc[0]
    first_comments.append(first_comment)
first_comments = pd.concat(first_comments, axis=1).T

# Make up counts for viz
first_responder_counts = first_comments.groupby(['org', 'author', 'authorAssociation'], as_index=False).    count().rename(columns={'id': 'n_first_responses'}).sort_values(['org', 'n_first_responses'], ascending=False)


# In[29]:


n_plot = 50

title = f"Top {n_plot} first responders for {github_org} in the last {n_days} days"
idata = first_responder_counts.groupby('author', as_index=False).agg({'n_first_responses': 'sum', 'authorAssociation': 'first'})
idata = idata.sort_values('n_first_responses', ascending=False).head(n_plot)
ch = alt.Chart(data=idata.head(n_plot), title=title).mark_bar().encode(
    x='author',
    y='n_first_responses',
    color=alt.Color('authorAssociation', scale=alt.Scale(domain=author_types, range=author_colors))
)
ch


# ## Recent activity
# 
# ### A list of merged PRs by project
# 
# Below is a tabbed readout of recently-merged PRs. Check out the title to get an idea for what they
# implemented, and be sure to thank the PR author for their hard work!

# In[30]:


tabs = widgets.Tab(children=[])

for ii, ((org, repo), imerged) in enumerate(merged.query("repo in @use_repos").groupby(['org', 'repo'])):
    merged_by = {}
    pr_by = {}
    issue_md = []
    issue_md.append(f"#### Closed PRs for repo: [{org}/{repo}](https://github.com/{github_org}/{repo})")
    issue_md.append("")
    issue_md.append(f"##### ")

    for _, ipr in imerged.iterrows():
        user_name = ipr['author']
        user_url = author_url(user_name)
        pr_number = ipr['number']
        pr_html = ipr['url']
        pr_title = ipr['title']
        pr_closedby = ipr['mergedBy']
        pr_closedby_url = f"https://github.com/{pr_closedby}"
        if user_name not in pr_by:
            pr_by[user_name] = 1
        else:
            pr_by[user_name] += 1

        if pr_closedby not in merged_by:
            merged_by[pr_closedby] = 1
        else:
            merged_by[pr_closedby] += 1
        text = f"* [(#{pr_number})]({pr_html}): _{pr_title}_ by **[@{user_name}]({user_url})** merged by **[@{pr_closedby}]({pr_closedby_url})**"
        issue_md.append(text)
    
    issue_md.append('')
    markdown_html = markdown('\n'.join(issue_md))

    children = list(tabs.children)
    children.append(HTML(markdown_html))
    tabs.children = tuple(children)
    tabs.set_title(ii, repo)
tabs


# ### A list of recent issues
# 
# Below is a list of issues with recent activity in each repository. If they seem of interest
# to you, click on their links and jump in to participate!

# In[31]:


# Add comment count data to issues and PRs
comment_counts = (
    comments
    .query("createdAt > @start_date and createdAt < @stop_date")
    .groupby(['org', 'repo', 'id'])
    .count().iloc[:, 0].to_frame()
)
comment_counts.columns = ['n_comments']
comment_counts = comment_counts.reset_index()


# In[32]:


n_plot = 5
tabs = widgets.Tab(children=[])

for ii, (repo, i_issues) in enumerate(comment_counts.query("repo in @use_repos").groupby('repo')):
    
    issue_md = []
    issue_md.append("")
    issue_md.append(f"##### [{github_org}/{repo}](https://github.com/{github_org}/{repo})")

    top_issues = i_issues.sort_values('n_comments', ascending=False).head(n_plot)
    top_issue_list = pd.merge(issues, top_issues, left_on=['org', 'repo', 'id'], right_on=['org', 'repo', 'id'])
    for _, issue in top_issue_list.sort_values('n_comments', ascending=False).head(n_plot).iterrows():
        user_name = issue['author']
        user_url = author_url(user_name)
        issue_number = issue['number']
        issue_html = issue['url']
        issue_title = issue['title']

        text = f"* [(#{issue_number})]({issue_html}): _{issue_title}_ by **[@{user_name}]({user_url})**"
        issue_md.append(text)

    issue_md.append('')
    md_html = HTML(markdown('\n'.join(issue_md)))

    children = list(tabs.children)
    children.append(HTML(markdown('\n'.join(issue_md))))
    tabs.children = tuple(children)
    tabs.set_title(ii, repo)
    
display(Markdown(f"Here are the top {n_plot} active issues in each repository in the last {n_days} days"))
display(tabs)


# # People reports

# In[33]:


#people = [
#    "choldgraf",
#    "betatim",
#    "GeorgianaElena",
#    "yuvipanda",
#    "consideRatio",
#    "chrisjsewell",
#    "AakashGfude",
#    "mmcky",
#]


# In[34]:


#def author_url(author):
#    return f"https://github.com/{author}"


# In[35]:


# Parameters
#fmt_date = "{:%Y-%m-%d}"

#n_days = 30 * 2
#start_date = fmt_date.format(pd.datetime.today() - timedelta(days=n_days))
#end_date = fmt_date.format(pd.datetime.today())

#renderer = "html"
#person = "jasongrout"


# In[36]:


#alt.renderers.enable(renderer);
#alt.themes.register('my_theme', alt_theme)
#alt.themes.enable("my_theme")


# In[37]:


#from pathlib import Path
#path_data = Path("./")
#comments = pd.read_csv(path_data.joinpath('../data/comments.csv'), index_col=0)
#issues = pd.read_csv(path_data.joinpath('../data/issues.csv'), index_col=0)
#prs = pd.read_csv(path_data.joinpath('../data/prs.csv'), index_col=0)

#comments = comments.query('author == @person').drop_duplicates()
#issues = issues.query('author == @person').drop_duplicates()
#closed_by = prs.query('mergedBy == @person')
#prs = prs.query('author == @person').drop_duplicates()


# In[38]:


# Time columns
# Also drop dates outside of our range
#time_columns = ['updatedAt', 'createdAt', 'closedAt']
#for col in time_columns:
#    for item in [comments, issues, prs, closed_by]:
#        if col not in item.columns:
#            continue
#        dt = pd.to_datetime(item[col]).dt.tz_localize(None)
#        item[col] = dt
#        item.query("updatedAt < @end_date and updatedAt > @start_date", inplace=True)


# In[39]:


#summaries = []
#for idata, name in [(issues, 'issues'), (prs, 'prs'), (comments, 'comments')]:
#    idata = idata.groupby(["repo", "org"]).agg({'id': "count"}).reset_index().rename(columns={'id': 'count'})
#    idata["kind"] = name
#    summaries.append(idata)
#summaries = pd.concat(summaries)


# In[40]:


#repo_summaries = summaries.groupby(["repo", "kind"]).agg({"count": "sum"}).reset_index()
#org_summaries = summaries.groupby(["org", "kind"]).agg({"count": "sum"}).reset_index()


# In[41]:


#ch1 = alt.Chart(repo_summaries, width=600, title="Activity per repository").mark_bar().encode(
#    x='repo',
#    y='count',
#    color='kind',
#    tooltip="kind"
#)

#ch2 = alt.Chart(repo_summaries, width=600, title="Log activity per repository").mark_bar().encode(
#    x='repo',
#    y='logcount',
#    color='kind',
#    tooltip="kind"
#)

#ch1 | ch2


# In[42]:


#alt.Chart(org_summaries, width=600).mark_bar().encode(
#    x='org',
#    y='count',
#    color='kind',
#    tooltip="org"
#)


# ## By repository over time

# ### Comments

# In[43]:


#comments_time = comments.groupby('repo').resample('W', on='createdAt').count()['author'].reset_index()
#comments_time = comments_time.rename(columns={'author': 'count'})
#comments_time_total = comments_time.groupby('createdAt').agg({"count": "sum"}).reset_index()
#ch1 = alt.Chart(comments_time, width=600).mark_line().encode(
#    x='createdAt',
#    y='count',
#    color='repo',
#    tooltip="repo"
#)

#ch2 = alt.Chart(comments_time_total, width=600).mark_line(color="black").encode(
#    x='createdAt',
#    y='count',
#)

#ch1 + ch2


# ### PRs

# In[44]:


#prs_time = prs.groupby('repo').resample('W', on='createdAt').count()['author'].reset_index()
#prs_time = prs_time.rename(columns={'author': 'count'})
#prs_time_total = prs_time.groupby('createdAt').agg({"count": "sum"}).reset_index()

#ch1 = alt.Chart(prs_time, width=600).mark_line().encode(
#    x='createdAt',
#    y='count',
#    color='repo',
#    tooltip="repo"
#)

#ch2 = alt.Chart(prs_time_total, width=600).mark_line(color="black").encode(
#    x='createdAt',
#    y='count',
#)

#ch1 + ch2


# In[45]:


#closed_by_time = closed_by.groupby('repo').resample('W', on='closedAt').count()['author'].reset_index()
#closed_by_time = closed_by_time.rename(columns={'author': 'count'})

#alt.Chart(closed_by_time, width=600).mark_line().encode(
#    x='closedAt',
#    y='count',
#    color='repo',
#    tooltip="repo"
#)


# ## By type over time

# In[46]:


#prs_time = prs[['author', 'createdAt']].resample('W', on='createdAt').count()['author'].reset_index()
#prs_time = prs_time.rename(columns={'author': 'prs'})
#comments_time = comments[['author', 'createdAt']].resample('W', on='createdAt').count()['author'].reset_index()
#comments_time = comments_time.rename(columns={'author': 'comments'})

#total_time = pd.merge(prs_time, comments_time, on='createdAt', how='outer')
#total_time = total_time.melt(id_vars='createdAt', var_name="kind", value_name="count")


# In[47]:


#alt.Chart(total_time, width=600).mark_line().encode(
#    x='createdAt',
#    y='count',
#    color='kind'
#)


# In[ ]:




