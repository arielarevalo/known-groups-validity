#!/bin/sh
# Check the paper after a build: TODO markers, missing numbers, undefined citations or
# references, and the page count. Exits non-zero when an item needs a fix.
cd "$(dirname "$0")/../../paper" || exit 1
LOG=build/main.log
status=0

todos=$(grep -n '\\TODO{' main.tex sections/*.tex | grep -v ':[[:space:]]*%' | wc -l | tr -d ' ')
if [ "$todos" -gt 0 ]; then
  echo "TODO markers left: $todos"
  grep -n '\\TODO{' main.tex sections/*.tex | grep -v ':[[:space:]]*%' | sed 's/^/  /' | cut -c1-110
  status=1
fi

missing=$(grep -o 'cv key missing: [^ ]*' "$LOG" 2>/dev/null | sort -u)
if [ -n "$missing" ]; then
  echo "Numbers missing from generated/numbers.tex:"; echo "$missing" | sed 's/^/  /'
  status=1
fi

undef=$(grep -E "(Citation|Reference) .* undefined" "$LOG" 2>/dev/null | sort -u)
if [ -n "$undef" ]; then
  echo "Undefined citations or references:"; echo "$undef" | sed 's/^/  /'
  status=1
fi

pages=$(sed -n 's/.*Output written on [^(]*(\([0-9]*\) page.*/\1/p' "$LOG" | tail -1)
echo "Pages in the PDF: ${pages:-unknown}, all pages."
mainend=$(sed -n 's/.*newlabel{endofmain}{{[^}]*}{\([0-9]*\)}.*/\1/p' build/main.aux 2>/dev/null)
backend=$(sed -n 's/.*newlabel{endofbackmatter}{{[^}]*}{\([0-9]*\)}.*/\1/p' build/main.aux 2>/dev/null)
echo "Main text ends on page ${mainend:-unknown}. Back matter ends on page ${backend:-unknown}."

[ $status -eq 0 ] && echo "check: OK" || echo "check: not ready"
exit $status
