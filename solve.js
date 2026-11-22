// Sokoban solver for the make.fun hallway-and-rooms generator (agus-ite entry).
//
// The generator lays a one-wide hallway with small rooms hanging off it.
// A box pushed out of a room into the hallway can never be pushed back, so
// every box stays inside its own room and the level's push optimum P* is
// the sum of the rooms' optima. This solver exploits that structure: it
// locates the rooms, solves each one separately with a breadth-first
// search over pushes (push-optimal by construction), and joins the room
// solutions with plain walks through the hallway. A board that does not
// split into rooms is solved with one global push search instead.
//
// Two cheap, sound prunings keep the searches small: dead squares (a box
// there can never reach any goal, found by pulling boxes backwards from
// the goals) and frozen 2x2 blocks of walls and boxes that trap a box off
// its goal. Work is budgeted in searched states, never in seconds, and
// every choice breaks ties in one fixed order, so two runs of the same
// level always produce the same bytes.

const DIRS = [
  [1, 0, 'r', 'R'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [0, -1, 'u', 'U'],
];

// State budgets per search. Work, never wall-clock time.
const ROOM_BUDGET = 500000;
const GLOBAL_BUDGET = 1200000;

/** Parse XSB level text into a flat board. Cells are indexed y * w + x. */
export function parseXSB(text) {
  const rows = String(text).split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = rows.reduce((m, r) => Math.max(m, r.length), 0);
  const wall = new Uint8Array(w * h).fill(1);
  const goalSet = new Set();
  const boxList = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === '#') continue;
      const i = y * w + x;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goalSet.add(i);
      if (ch === '$' || ch === '*') boxList.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  // The board frame is solid: nothing ever leaves the rectangle.
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxList.sort((a, b) => a - b);
  return { w, h, wall, goals: [...goalSet].sort((a, b) => a - b), boxes: boxList, player };
}

/**
 * Live squares: every cell from which one box could still be pushed onto
 * some goal. Computed backwards from the goals — a cell is live when the
 * box can be pushed from it onto a live cell with floor behind it for the
 * player to stand on. A push that lands a box on a dead square is pruned.
 */
function liveSquares(board) {
  const { w, wall, goals } = board;
  const live = new Uint8Array(wall.length);
  const queue = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const from = cell - d;
      const stand = from - d;
      if (from < 0 || stand < 0 || from >= wall.length || stand >= wall.length) continue;
      if (wall[from] || wall[stand] || live[from]) continue;
      live[from] = 1;
      queue.push(from);
    }
  }
  return live;
}

/**
 * Flood-fill the cells the player can walk to without moving any box.
 * Returns the seen mask plus the smallest reachable cell id, which stands
 * for the whole region when two positions must compare equal.
 */
function floodRegion(board, boxMask, start) {
  const { w, wall } = board;
  const seen = new Uint8Array(wall.length);
  const stack = [start];
  seen[start] = 1;
  let min = start;
  while (stack.length > 0) {
    const cell = stack.pop();
    if (cell < min) min = cell;
    for (const [dx, dy] of DIRS) {
      const next = cell + dy * w + dx;
      if (next < 0 || next >= wall.length) continue;
      if (wall[next] || boxMask[next] || seen[next]) continue;
      seen[next] = 1;
      stack.push(next);
    }
  }
  return { seen, min };
}

/** Shortest walk (lowercase LURD) between two cells, or null if cut off. */
function shortestWalk(board, boxMask, from, to) {
  const { w, wall } = board;
  if (from === to) return '';
  const parent = new Int32Array(wall.length).fill(-2);
  const stepLetter = new Array(wall.length).fill('');
  const queue = [from];
  parent[from] = -1;
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    if (cell === to) break;
    for (const [dx, dy, walk] of DIRS) {
      const next = cell + dy * w + dx;
      if (next < 0 || next >= wall.length) continue;
      if (wall[next] || boxMask[next] || parent[next] !== -2) continue;
      parent[next] = cell;
      stepLetter[next] = walk;
      queue.push(next);
    }
  }
  if (parent[to] === -2) return null;
  let path = '';
  for (let cell = to; cell !== from; cell = parent[cell]) path = stepLetter[cell] + path;
  return path;
}

/**
 * True when the box that just landed on `target` completes a solid 2x2
 * block of walls and boxes while at least one box in the block is off its
 * goal. Nothing inside such a block can ever move again.
 */
function frozenBlock(board, boxMask, goalMask, target) {
  const { w, wall } = board;
  const anchors = [target, target - 1, target - w, target - w - 1];
  for (const a of anchors) {
    if (a < 0 || a + w + 1 >= wall.length) continue;
    const cells = [a, a + 1, a + w, a + w + 1];
    if (!cells.every((c) => wall[c] || boxMask[c])) continue;
    if (cells.some((c) => boxMask[c] && !goalMask[c])) return true;
  }
  return false;
}

/**
 * Breadth-first search over pushes on one board. A state is the sorted
 * list of box cells plus the player's reachable region reduced to its
 * smallest cell, so positions that differ only in where the player stands
 * inside one region merge into a single state. States expand in push
 * order, hence the first solved state uses the fewest pushes. Returns
 * { pushes, moves } with moves in LURD (uppercase marks a push), or null
 * when the state budget runs out first.
 */
function pushSearch(board, budget) {
  const { w, wall, goals } = board;
  const n = wall.length;
  const goalMask = new Uint8Array(n);
  for (const g of goals) goalMask[g] = 1;
  const live = liveSquares(board);
  if (board.boxes.some((b) => !live[b])) return null;
  const isSolved = (boxes) => boxes.every((b) => goalMask[b] === 1);

  const boxMask = new Uint8Array(n);
  const startBoxes = board.boxes.slice().sort((a, b) => a - b);
  const nodes = [{ boxes: startBoxes, player: board.player, parent: -1 }];
  const visited = new Set();
  for (const b of startBoxes) boxMask[b] = 1;
  visited.add(startBoxes.join('#') + '@' + floodRegion(board, boxMask, board.player).min);
  for (const b of startBoxes) boxMask[b] = 0;
  let found = isSolved(startBoxes) ? 0 : -1;

  for (let head = 0; found < 0 && head < nodes.length; head++) {
    const node = nodes[head];
    const boxes = node.boxes;
    for (const b of boxes) boxMask[b] = 1;
    const region = floodRegion(board, boxMask, node.player).seen;
    for (let i = 0; i < boxes.length && found < 0; i++) {
      const box = boxes[i];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const stand = box - d;
        const dest = box + d;
        if (stand < 0 || stand >= n || dest < 0 || dest >= n) continue;
        if (!region[stand] || wall[dest] || boxMask[dest] || !live[dest]) continue;
        boxMask[box] = 0;
        boxMask[dest] = 1;
        const frozen = frozenBlock(board, boxMask, goalMask, dest);
        let nextBoxes = null;
        let key = '';
        if (!frozen) {
          nextBoxes = boxes.slice();
          nextBoxes[i] = dest;
          nextBoxes.sort((a, b) => a - b);
          key = nextBoxes.join('#') + '@' + floodRegion(board, boxMask, box).min;
        }
        boxMask[dest] = 0;
        boxMask[box] = 1;
        if (frozen || visited.has(key)) continue;
        if (nodes.length >= budget) {
          for (const b of boxes) boxMask[b] = 0;
          return null;
        }
        visited.add(key);
        nodes.push({ boxes: nextBoxes, player: box, parent: head });
        if (isSolved(nextBoxes)) found = nodes.length - 1;
      }
    }
    for (const b of boxes) boxMask[b] = 0;
  }
  if (found < 0) return null;

  // Walk the parent chain back, then replay it forward stitching the
  // player's walks between consecutive pushes.
  const chain = [];
  for (let at = found; at > 0; at = nodes[at].parent) chain.push(at);
  chain.reverse();
  let moves = '';
  let player = board.player;
  let boxes = startBoxes;
  for (const at of chain) {
    const nextBoxes = nodes[at].boxes;
    const nextSet = new Set(nextBoxes);
    const boxSet = new Set(boxes);
    const from = boxes.find((b) => !nextSet.has(b));
    const to = nextBoxes.find((b) => !boxSet.has(b));
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d);
    for (const b of boxes) boxMask[b] = 1;
    const walk = shortestWalk(board, boxMask, player, from - d);
    for (const b of boxes) boxMask[b] = 0;
    if (walk === null || !dir) return null;
    moves += walk + dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: chain.length, moves };
}

/** Where the player ends up after playing a move string. */
function playerAfter(w, start, moves) {
  let cell = start;
  for (const ch of moves.toLowerCase()) {
    if (ch === 'u') cell -= w;
    else if (ch === 'd') cell += w;
    else if (ch === 'l') cell -= 1;
    else cell += 1;
  }
  return cell;
}

/**
 * Split the board into rooms. The hallway is the player's row; cutting
 * away its dead squares leaves one connected piece of floor per room
 * (plus the hallway's live stretches). Returns the rooms that hold boxes,
 * or null when the board is not a set of rooms — a box sitting on a cut
 * square, or a piece whose box and goal counts disagree — in which case
 * the caller falls back to solving the whole board at once.
 */
function findRooms(board) {
  const { w, wall, goals, boxes, player } = board;
  const live = liveSquares(board);
  const hallwayRow = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let c = hallwayRow * w; c < (hallwayRow + 1) * w; c++) {
    if (!wall[c] && !live[c]) cut[c] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const partOf = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let seed = 0; seed < wall.length; seed++) {
    if (wall[seed] || cut[seed] || partOf[seed] >= 0) continue;
    const cells = [seed];
    partOf[seed] = rooms.length;
    for (let i = 0; i < cells.length; i++) {
      for (const [dx, dy] of DIRS) {
        const next = cells[i] + dy * w + dx;
        if (next < 0 || next >= wall.length) continue;
        if (wall[next] || cut[next] || partOf[next] >= 0) continue;
        partOf[next] = rooms.length;
        cells.push(next);
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[partOf[b]].boxes.push(b);
  for (const g of goals) {
    if (partOf[g] < 0) return null;
    rooms[partOf[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * Solve room by room. While one room is being searched, the other rooms'
 * boxes (their goals once finished) count as walls, which keeps each
 * search small and the rooms independent. The player walks on from
 * wherever the previous room left it.
 */
function solveByRooms(board) {
  const rooms = findRooms(board);
  if (rooms === null) return pushSearch(board, GLOBAL_BUDGET);
  let player = board.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = board.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const c of other.done ? other.goals : other.boxes) wall[c] = 1;
    }
    const sub = {
      w: board.w,
      h: board.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const result = pushSearch(sub, ROOM_BUDGET);
    if (result === null) return null;
    pushes += result.pushes;
    moves += result.moves;
    player = playerAfter(board.w, player, result.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/**
 * Solve one XSB level and return its LURD move string. An empty string
 * means no solution was found within budget; it is still valid format.
 * Never throws.
 */
export function solveLevel(text) {
  try {
    const board = parseXSB(text);
    if (board.player < 0 || board.boxes.length === 0) return '';
    if (board.boxes.length !== board.goals.length) return '';
    const byRooms = solveByRooms(board);
    if (byRooms !== null) return byRooms.moves;
    const whole = pushSearch(board, GLOBAL_BUDGET);
    return whole !== null ? whole.moves : '';
  } catch {
    return '';
  }
}

export const solve = solveLevel;

