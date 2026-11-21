# Learnings

A running collection of notes from courses, books, talks, and things I figure out on my own.

## Structure

| Folder | What goes here |
| --- | --- |
| `courses/` | One folder per course, e.g. `courses/kubernetes-deep-dive/` |
| `notes/` | Standalone learnings that don't belong to a course (one topic per file) |
| `inbox/` | Quick captures, unfiled. Empty this into `courses/` or `notes/` regularly |
| `templates/` | Starting points for a new course, lecture, or standalone note |

## Conventions

- File names are lowercase with dashes: `01-intro-to-etcd.md`
- Lecture notes start with a number so they sort in the order I went through them
- Every course folder has a `README.md` that links to its notes
- Write notes in my own words. If I'm copying something, put it in a quote block and link the source
- Date anything time-sensitive with `YYYY-MM-DD`

## Adding a new course

```sh
mkdir -p courses/my-new-course
cp templates/course-readme.md courses/my-new-course/README.md
mkdir -p courses/my-new-course/notes
```

Then start writing notes from `templates/lecture.md`.

## Adding a standalone learning

Copy `templates/note.md` into `notes/` and rename it to the topic.

## Index

### Courses

- _none yet_

### Notes

- _none yet_
