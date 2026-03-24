import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import axios from 'axios'
import './App.css'
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd'

// ─── ScoreInput ───────────────────────────────────────────────────────────────
function ScoreInput({ album, initialValue, onSave, onCancel }) {
  const [val, setVal] = useState(initialValue ?? '')

  const commit = () => {
    const num = parseInt(val.trim(), 10)
    if (!isNaN(num) && num >= 0 && num <= 100) {
      onSave(album, num, null)
    } else {
      onCancel()
    }
  }

  return (
    <input
      type="text"
      value={val}
      onChange={e => setVal(e.target.value)}
      onBlur={commit}
      onKeyDown={e => {
        if (e.key === 'Enter') { e.preventDefault(); commit() }
        if (e.key === 'Escape') onCancel()
      }}
      autoFocus
      style={{ width: '70px', textAlign: 'center' }}
    />
  )
}

// ─── NotesEditor ──────────────────────────────────────────────────────────────
function NotesEditor({ album, onSave }) {
  const [editing, setEditing] = useState(false)
  const [val, setVal] = useState(album.notes || '')

  const commit = () => {
    setEditing(false)
    if (val !== (album.notes || '')) {
      onSave(album, null, val)
    }
  }

  return editing ? (
    <input
      type="text"
      value={val}
      onChange={e => setVal(e.target.value)}
      onBlur={commit}
      onKeyDown={e => {
        if (e.key === 'Enter') commit()
        if (e.key === 'Escape') { setVal(album.notes || ''); setEditing(false) }
      }}
      autoFocus
      placeholder="Add a note..."
      style={{
        width: '100%',
        fontStyle: 'italic',
        fontSize: '0.8rem',
        color: '#777',
        border: 'none',
        borderBottom: '1px solid #ccc',
        outline: 'none',
        background: 'transparent',
        marginTop: '0.5rem',
        padding: '2px 0',
      }}
    />
  ) : (
    <div
      onClick={() => setEditing(true)}
      style={{
        cursor: 'pointer',
        fontStyle: 'italic',
        fontSize: '0.8rem',
        color: val ? '#777' : '#ccc',
        marginTop: '0.5rem',
        minHeight: '1.2rem',
      }}
    >
      {val ? `"${val}"` : '+ add note'}
    </div>
  )
}

// ─── Feed type badge ──────────────────────────────────────────────────────────
const FEED_BADGES = {
  new_potential: { label: '✨ New Potential', color: '#1DB954' },
  take_a_chance: { label: '🎰 Take a Chance', color: '#f59e0b' },
  old_fav:       { label: '⭐ Old Fav',        color: '#6366f1' },
}

// ─── YearBadge ────────────────────────────────────────────────────────────────
const YEAR_BADGE_STYLES = {
  '2026': { background: '#fdf6e3', color: '#c1440e', border: '1px solid #e8c9a0' },
  '2025': { background: '#2d3748', color: '#e2e8f0', border: '1px solid #4a5568' },
  default: { background: '#f0f0f0', color: '#666', border: '1px solid #ddd' },
}

function YearBadge({ year }) {
  if (!year) return null
  const style = YEAR_BADGE_STYLES[year] || YEAR_BADGE_STYLES.default
  return (
    <span style={{
      ...style,
      fontSize: '0.7rem',
      fontWeight: 700,
      borderRadius: '6px',
      padding: '2px 7px',
      letterSpacing: '0.04em',
      display: 'inline-block',
      marginBottom: '0.4rem',
    }}>
      {year}
    </span>
  )
}

// ─── AlbumCard ────────────────────────────────────────────────────────────────
function AlbumCard({
  album,
  editingId,
  coverErrors,
  coverFixExpanded,
  getCoverUrl,
  onCoverError,
  onStartEdit,
  onSaveScore,
  onNuke,
  onCancelEdit,
  onSaveCover,
  onToggleCoverFix,
}) {
  const albumKey = `${album.Artist}|${album.Album}`
  const coverUrl = getCoverUrl(album)
  const hasCoverError = coverErrors[albumKey]

  const gut = album.gut_score
  const pred = album.avg_score
  const scoreColor = (g, p) => {
    if (g === undefined || g === null) return 'inherit'
    if (g > p) return '#2e7d32'
    if (g < p) return '#1976d2'
    return '#757575'
  }

  const badge = album.feed_type ? FEED_BADGES[album.feed_type] : null
  const rawSpotifyUrl = album['Spotify URL'] || null
  const spotifyHref = rawSpotifyUrl
    ? (rawSpotifyUrl.startsWith('http') ? rawSpotifyUrl : `https://${rawSpotifyUrl}`)
    : null

  // ── Swipe-to-cut state ──
  const dragStartX = useRef(null)
  const dragStartY = useRef(null)
  const isDragging = useRef(false)
  const isHorizontal = useRef(null)
  const [dragX, setDragX] = useState(0)
  const [phase, setPhase] = useState('idle') // idle | dragging | flying | collapsing
  const [cardHeight, setCardHeight] = useState(null)
  const cardRef = useRef(null)
  const THRESHOLD = 120

  const getClientX = (e) => e.touches ? e.touches[0].clientX : e.clientX
  const getClientY = (e) => e.touches ? e.touches[0].clientY : e.clientY

  const onDragStart = (e) => {
    if (e.button === 2) return // ignore right click
    dragStartX.current = getClientX(e)
    dragStartY.current = getClientY(e)
    isDragging.current = true
    isHorizontal.current = null
  }

  const onDragMove = (e) => {
    if (!isDragging.current) return
    const dx = getClientX(e) - dragStartX.current
    const dy = getClientY(e) - dragStartY.current

    // Lock axis on first significant move
    if (isHorizontal.current === null) {
      if (Math.abs(dx) > 6 || Math.abs(dy) > 6) {
        isHorizontal.current = Math.abs(dx) > Math.abs(dy)
      }
      return
    }

    if (!isHorizontal.current) return // vertical scroll, bail

    if (dx < 0) {
      e.preventDefault()
      setPhase('dragging')
      setDragX(dx)
    }
  }

  const onDragEnd = () => {
    if (!isDragging.current) return
    isDragging.current = false
    isHorizontal.current = null

    if (dragX < -THRESHOLD) {
      // Capture height before flying off
      if (cardRef.current) setCardHeight(cardRef.current.offsetHeight)
      setPhase('flying')
      setTimeout(() => {
        setPhase('collapsing')
        setTimeout(() => {
          onNuke(album)
        }, 320)
      }, 200)
    } else {
      setPhase('idle')
      setDragX(0)
    }
  }

  const cutOpacity = Math.min(1, Math.abs(dragX) / THRESHOLD)
  const rotation = dragX * 0.03

  const cardStyle = (() => {
    if (phase === 'idle') return { transform: 'translateX(0)', transition: 'transform 0.3s cubic-bezier(0.34,1.56,0.64,1)' }
    if (phase === 'dragging') return { transform: `translateX(${dragX}px) rotate(${rotation}deg)`, transition: 'none' }
    if (phase === 'flying') return { transform: 'translateX(-110vw) rotate(-8deg)', transition: 'transform 0.2s ease-in' }
    if (phase === 'collapsing') return { transform: 'translateX(-110vw)', transition: 'none' }
    return {}
  })()

  const wrapperStyle = (() => {
    if (phase === 'collapsing') return {
      height: `${cardHeight}px`,
      overflow: 'hidden',
      animation: 'collapseCard 0.32s ease-out forwards',
    }
    return { position: 'relative' }
  })()

  return (
    <div style={wrapperStyle}>
      {/* Cut zone behind card */}
      <div style={{
        position: 'absolute',
        inset: 0,
        background: `rgba(211,47,47,${cutOpacity * 0.85})`,
        borderRadius: '12px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        paddingRight: '1.5rem',
        pointerEvents: 'none',
      }}>
        <span style={{
          color: '#fff',
          fontWeight: 700,
          fontSize: '1rem',
          letterSpacing: '0.05em',
          opacity: cutOpacity,
          transform: `scale(${0.8 + cutOpacity * 0.2})`,
          transition: phase === 'dragging' ? 'none' : 'all 0.3s',
        }}>
          CUT
        </span>
      </div>

      <div
        ref={cardRef}
        className="album-card"
        style={{ ...cardStyle, position: 'relative', cursor: phase === 'dragging' ? 'grabbing' : 'default', userSelect: 'none' }}
        onMouseDown={onDragStart}
        onMouseMove={onDragMove}
        onMouseUp={onDragEnd}
        onMouseLeave={onDragEnd}
        onTouchStart={onDragStart}
        onTouchMove={onDragMove}
        onTouchEnd={onDragEnd}
      >
        <div
          className="card-cover"
          style={{ position: 'relative', cursor: 'pointer' }}
          onDoubleClick={() => {
            const query = encodeURIComponent(`${album.Artist} ${album.Album} album cover`)
            window.open(`https://www.google.com/search?q=${query}&tbm=isch`, '_blank')
            onToggleCoverFix(albumKey)
          }}
        >
          {coverUrl && !hasCoverError ? (
            <img
              src={coverUrl}
              alt="cover"
              onError={() => onCoverError(albumKey)}
              style={{ width: '100%', height: 'auto', borderRadius: '8px', pointerEvents: 'none' }}
            />
          ) : (
            <div style={{ width: '100%', aspectRatio: '1/1', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f0f0f0', borderRadius: '8px', fontSize: '3rem' }}>🎵</div>
          )}
          {coverFixExpanded[albumKey] && (
            <div
              onMouseDown={e => e.stopPropagation()}
              style={{ position: 'absolute', bottom: 0, left: 0, right: 0, background: 'rgba(0,0,0,0.85)', padding: '0.5rem', borderRadius: '0 0 8px 8px' }}
            >
              <input
                type="text"
                placeholder="Paste image URL..."
                id={`cover-url-${albumKey}`}
                onDoubleClick={e => e.stopPropagation()}
                style={{ width: '100%', fontSize: '0.75rem', padding: '4px', borderRadius: '4px', border: 'none', marginBottom: '4px' }}
              />
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  const url = document.getElementById(`cover-url-${albumKey}`).value
                  if (url) onSaveCover(album, url)
                }}
                style={{ width: '100%', background: '#1DB954', color: '#fff', border: 'none', borderRadius: '4px', padding: '4px', cursor: 'pointer', fontSize: '0.75rem' }}
              >
                Save
              </button>
            </div>
          )}
        </div>

        <div className="card-content">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '0.4rem' }}>
            <YearBadge year={album.prediction_year} />
            {album.feed_type === 'old_fav' && album.vault_rank && album.vault_year ? (
              <span style={{
                fontSize: '0.65rem',
                fontWeight: 600,
                color: '#4a6fa5',
                background: '#eef2f8',
                border: '1px solid #c5d3e8',
                borderRadius: '6px',
                padding: '2px 7px',
                letterSpacing: '0.03em',
              }}>
                #{album.vault_rank} · {album.vault_year}
              </span>
            ) : badge && album.feed_type !== 'old_fav' ? (
              <span style={{ fontSize: '0.65rem', fontWeight: 500, color: '#aaa', letterSpacing: '0.03em' }}>
                {badge.label.replace(/^[^\w]+/, '')}
              </span>
            ) : null}
          </div>

          <div className="card-title">
            <span className="artist">{album.Artist}</span> –{' '}
            <span className="album">{album.Album}</span>
          </div>

          <div className="card-scores">
            <div className="predicted-score">
              Predicted: {album.avg_score?.toFixed(1) || '—'}
            </div>
            <div className="gut-score">
              Your Score:{' '}
              {editingId === albumKey ? (
                <ScoreInput
                  album={album}
                  initialValue={gut !== undefined && gut !== null ? gut.toString() : ''}
                  onSave={onSaveScore}
                  onCancel={onCancelEdit}
                />
              ) : (
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                  <span
                    title="Click to rate"
                    onClick={() => onStartEdit(album)}
                    style={{
                      cursor: 'pointer',
                      fontWeight: 'bold',
                      display: 'inline-block',
                      minWidth: '40px',
                      color: scoreColor(gut, pred),
                    }}
                  >
                    {gut !== undefined && gut !== null ? gut : '—'}
                  </span>
                  {gut !== undefined && gut !== null && pred !== undefined && (
                    <span
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: 'normal',
                        color: scoreColor(gut, pred),
                        backgroundColor: '#f0f0f0',
                        padding: '2px 4px',
                        borderRadius: '12px',
                      }}
                    >
                      {(() => {
                        const delta = gut - pred
                        return delta > 0 ? `+${Math.round(delta)}` : `${Math.round(delta)}`
                      })()}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          {album.Genres && (
            <div className="card-metadata"><strong>Genres:</strong> {album.Genres}</div>
          )}
          {album['Similar Artists'] && (
            <div className="card-metadata"><strong>Similar Artists:</strong> {album['Similar Artists']}</div>
          )}
          {album['Record Label'] && (
            <div className="card-metadata"><strong>Label:</strong> {album['Record Label']}</div>
          )}
          {album['Release Date'] && (
            <div className="card-metadata"><strong>Released:</strong> {album['Release Date']}</div>
          )}
					{spotifyHref && (
						<a
							href={spotifyHref}
							target="_blank"
							rel="noopener noreferrer"
							className="spotify-link"
						>
							▶ Play on Spotify
						</a>
					)}

          <NotesEditor album={album} onSave={onSaveScore} />

          {!coverUrl && (
            <div className="cover-fixer">
              <button onClick={() => onToggleCoverFix(albumKey)}>🖼️ Fix Cover</button>
              {coverFixExpanded[albumKey] && (
                <div className="cover-fixer-panel">
                  <p>Paste image URL:</p>
                  <input
                    type="text"
                    placeholder="https://..."
                    id={`cover-url-${albumKey}`}
                    style={{ width: '100%', marginBottom: '0.5rem' }}
                  />
                  <button
                    onClick={() => {
                      const url = document.getElementById(`cover-url-${albumKey}`).value
                      if (url) onSaveCover(album, url)
                    }}
                  >
                    💾 Save Cover
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────
function App() {
  const [activeTab, setActiveTab] = useState('weekly')
  const [weeks, setWeeks] = useState([])
  const [selectedWeek, setSelectedWeek] = useState('')
  const [albums, setAlbums] = useState([])
  const [loading, setLoading] = useState(false)
  const [editingId, setEditingId] = useState(null)
  const [coverErrors, setCoverErrors] = useState({})
  const [top100, setTop100] = useState([])
  const [orderedTop100, setOrderedTop100] = useState([])
  const [top100Loading, setTop100Loading] = useState(false)
  const [coverFixExpanded, setCoverFixExpanded] = useState({})

  // Vault state
  const [vaultAlbums, setVaultAlbums] = useState([])
  const [vaultLoading, setVaultLoading] = useState(false)
  const [vaultYear, setVaultYear] = useState(2025)

  // Discover feed state
  const [feedAlbums, setFeedAlbums] = useState([])
  const [feedLoading, setFeedLoading] = useState(false)
  const [feedOffset, setFeedOffset] = useState(0)
  const [feedHasMore, setFeedHasMore] = useState(true)

  const sortedAlbums = useMemo(() => {
    return [...albums].sort((a, b) => {
      const aVal = a.gut_score ?? a.avg_score ?? 0
      const bVal = b.gut_score ?? b.avg_score ?? 0
      return bVal - aVal
    })
  }, [albums])

  const getCoverUrl = (album) => {
    if (!album['Album Art']) return null
    let url = album['Album Art'].replace(/\\/g, '/')
    if (url.startsWith('/covers/')) url = url.substring(1)
    else if (url.startsWith('./covers/')) url = url.substring(2)
    if (url.startsWith('covers/')) return `http://localhost:8000/${url}`
    return url
  }

  const handleCoverError = (albumKey) => {
    setCoverErrors(prev => ({ ...prev, [albumKey]: true }))
  }

  useEffect(() => {
    axios.get('http://localhost:8000/weeks')
      .then(res => {
        setWeeks(res.data)
        if (res.data.length > 0) setSelectedWeek(res.data[0].formatted)
      })
      .catch(err => console.error('Failed to fetch weeks', err))
  }, [])

  useEffect(() => {
    if (!selectedWeek) return
    setLoading(true)
    axios.get(`http://localhost:8000/albums/${encodeURIComponent(selectedWeek)}`)
      .then(res => setAlbums(res.data.albums))
      .catch(err => console.error('Failed to fetch albums', err))
      .finally(() => setLoading(false))
  }, [selectedWeek])

  useEffect(() => {
    if (activeTab !== 'top100') return
    setTop100Loading(true)
    Promise.all([
      axios.get('http://localhost:8000/top100'),
      axios.get('http://localhost:8000/top100/order'),
    ]).then(async ([res, orderRes]) => {
      const newList = res.data
      const savedOrder = orderRes.data
      const albumMap = new Map(newList.map(a => [`${a.Artist}|${a.Album}`, a]))
      const savedSet = new Set(savedOrder)
      const ordered = savedOrder.map(k => albumMap.get(k)).filter(Boolean)

      for (const album of newList) {
        const key = `${album.Artist}|${album.Album}`
        if (!savedSet.has(key)) {
          const score = album.gut_score ?? album.avg_score ?? 0
          let i = 0
          while (i < ordered.length && (ordered[i].gut_score ?? ordered[i].avg_score ?? 0) >= score) i++
          ordered.splice(i, 0, album)
        }
      }

      await axios.post('http://localhost:8000/top100/order', ordered.map(a => `${a.Artist}|${a.Album}`))
      setOrderedTop100(ordered)
      setTop100(newList)
    }).catch(err => console.error('Failed to fetch top 100', err))
      .finally(() => setTop100Loading(false))
  }, [activeTab])

  // Load vault when switching to vault tab
  useEffect(() => {
    if (activeTab !== 'vault') return
    if (vaultAlbums.length > 0) return
    setVaultLoading(true)
    axios.get('http://localhost:8000/vault')
      .then(res => setVaultAlbums(res.data))
      .catch(err => console.error('Failed to load vault', err))
      .finally(() => setVaultLoading(false))
  }, [activeTab])

  // Load initial feed when switching to discover tab
  useEffect(() => {
    if (activeTab !== 'discover') return
    if (feedAlbums.length > 0) return  // already loaded
    loadMoreFeed(0, true)
  }, [activeTab])

  const loadMoreFeed = async (offset, reset = false) => {
    setFeedLoading(true)
    try {
      const res = await axios.get(`http://localhost:8000/discover/feed?limit=20&offset=${offset}`)
      const newCards = res.data
      if (reset) {
        setFeedAlbums(newCards)
      } else {
        setFeedAlbums(prev => [...prev, ...newCards])
      }
      setFeedOffset(offset + 1)
      setFeedHasMore(newCards.length === 20)
    } catch (err) {
      console.error('Failed to load feed', err)
    } finally {
      setFeedLoading(false)
    }
  }

  const handleStartEdit = (album) => {
    setEditingId(`${album.Artist}|${album.Album}`)
  }

  const handleSaveScore = async (album, newScore, newNotes) => {
    const scoreToSave = newScore !== null && newScore !== undefined ? newScore : (album.gut_score ?? 0)
    const notesToSave = newNotes !== null && newNotes !== undefined ? newNotes : (album.notes || '')
    try {
      await axios.post('http://localhost:8000/rate', {
        artist: album.Artist,
        album: album.Album,
        score: scoreToSave,
        notes: notesToSave,
      })
      const updater = a =>
        a.Artist === album.Artist && a.Album === album.Album
          ? { ...a, gut_score: scoreToSave, notes: notesToSave }
          : a
      setAlbums(prev => prev.map(updater))
      setOrderedTop100(prev => prev.map(updater))
      // When rated in feed, remove card from feed (it's now rated)
      if (newScore !== null && newScore !== undefined) {
        setFeedAlbums(prev => prev.filter(a =>
          !(a.Artist === album.Artist && a.Album === album.Album)
        ))
      } else {
        setFeedAlbums(prev => prev.map(updater))
      }
    } catch (err) {
      console.error('Failed to save score', err)
    }
    if (newScore !== null) setEditingId(null)
  }

  const handleNuke = async (album) => {
    try {
      await axios.post('http://localhost:8000/nuke', {
        artist: album.Artist,
        album: album.Album,
      })
      setAlbums(prev => prev.filter(a =>
        !(a.Artist === album.Artist && a.Album === album.Album)
      ))
      setOrderedTop100(prev => prev.filter(t =>
        !(t.Artist === album.Artist && t.Album === album.Album)
      ))
      setFeedAlbums(prev => prev.filter(a =>
        !(a.Artist === album.Artist && a.Album === album.Album)
      ))
    } catch (err) {
      console.error('Failed to nuke album', err)
    }
    setEditingId(null)
  }

  const handleCancelEdit = () => setEditingId(null)

  const handleSaveCover = async (album, imageUrl) => {
    try {
      await axios.post('http://localhost:8000/save_cover', {
        artist: album.Artist,
        album: album.Album,
        image_url: imageUrl,
      })
      const updater = a =>
        a.Artist === album.Artist && a.Album === album.Album
          ? { ...a, 'Album Art': imageUrl }
          : a
      setAlbums(prev => prev.map(updater))
      setTop100(prev => prev.map(updater))
      setOrderedTop100(prev => prev.map(updater))
      setFeedAlbums(prev => prev.map(updater))
      setVaultAlbums(prev => prev.map(updater))
      setCoverFixExpanded(prev => ({ ...prev, [`${album.Artist}|${album.Album}`]: false }))
    } catch (err) {
      console.error('Failed to save cover', err)
    }
  }

  const handleToggleCoverFix = (albumKey) => {
    setCoverFixExpanded(prev => ({ ...prev, [albumKey]: !prev[albumKey] }))
  }

  const cardProps = {
    editingId,
    coverErrors,
    coverFixExpanded,
    getCoverUrl,
    onCoverError: handleCoverError,
    onStartEdit: handleStartEdit,
    onSaveScore: handleSaveScore,
    onNuke: handleNuke,
    onCancelEdit: handleCancelEdit,
    onSaveCover: handleSaveCover,
    onToggleCoverFix: handleToggleCoverFix,
  }

  const handleDragEnd = async (result) => {
    if (!result.destination) return
    const items = Array.from(orderedTop100)
    const [moved] = items.splice(result.source.index, 1)
    items.splice(result.destination.index, 0, moved)
    setOrderedTop100(items)
    try {
      await axios.post('http://localhost:8000/top100/order', items.map(i => `${i.Artist}|${i.Album}`))
    } catch (err) {
      console.error('Failed to save order', err)
    }
  }

  const Top100Table = () => {
    const [editingReviewId, setEditingReviewId] = useState(null)
    const [editReviewValue, setEditReviewValue] = useState('')

    const handleSaveReview = async (album, newNotes) => {
      try {
        await axios.post('http://localhost:8000/rate', {
          artist: album.Artist,
          album: album.Album,
          score: album.gut_score,
          notes: newNotes,
        })
        setOrderedTop100(prev => prev.map(t =>
          t.Artist === album.Artist && t.Album === album.Album ? { ...t, notes: newNotes } : t
        ))
        setEditingReviewId(null)
      } catch (err) {
        console.error('Failed to save review', err)
      }
    }

    if (top100Loading) return <p>Loading top 100...</p>
    if (!orderedTop100.length) return <p>No albums in top 100 yet.</p>

    return (
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="top100">
          {(provided) => (
            <div {...provided.droppableProps} ref={provided.innerRef} className="billboard">
              <div className="billboard-header">
                <div className="rank-col">#</div>
                <div className="cover-col">Cover</div>
                <div className="artist-col">Artist</div>
                <div className="album-col">Album</div>
                <div className="review-col">Review</div>
              </div>
              {orderedTop100.map((item, idx) => {
                const albumKey = `${item.Artist}|${item.Album}`
                const coverUrl = getCoverUrl(item)
                const hasCoverError = coverErrors[albumKey]
                return (
                  <Draggable key={albumKey} draggableId={albumKey} index={idx}>
                    {(provided) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                        className="billboard-row"
                      >
                        <div className="rank-col">{idx + 1}</div>
                        <div className="cover-col">
                          {coverUrl && !hasCoverError ? (
                            <img
                              src={coverUrl}
                              alt="cover"
                              width="40"
                              height="40"
                              style={{ objectFit: 'cover' }}
                              onError={() => handleCoverError(albumKey)}
                            />
                          ) : (
                            <div style={{ width: 40, height: 40, background: '#f0f0f0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>🎵</div>
                          )}
                        </div>
                        <div className="artist-col">{item.Artist}</div>
                        <div className="album-col">
                          {item['Spotify URL'] ? (
                            <a
                              href={item['Spotify URL'].startsWith('http') ? item['Spotify URL'] : `https://${item['Spotify URL']}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{ textDecoration: 'none', color: 'inherit' }}
                            >
                              {item.Album}
                            </a>
                          ) : item.Album}
                        </div>
                        <div className="review-col">
                          {editingReviewId === albumKey ? (
                            <input
                              type="text"
                              value={editReviewValue}
                              onChange={e => setEditReviewValue(e.target.value)}
                              onBlur={() => handleSaveReview(item, editReviewValue)}
                              onKeyDown={e => {
                                if (e.key === 'Enter') handleSaveReview(item, editReviewValue)
                                if (e.key === 'Escape') setEditingReviewId(null)
                              }}
                              autoFocus
                              style={{ width: '100%', padding: '2px 4px' }}
                            />
                          ) : (
                            <span
                              onClick={() => { setEditingReviewId(albumKey); setEditReviewValue(item.notes || '') }}
                              style={{ cursor: 'pointer' }}
                            >
                              {item.notes || '—'}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </Draggable>
                )
              })}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
    )
  }

  const VaultPage = ({ coverFixExpanded, onToggleCoverFix, onSaveCover, getCoverUrl }) => {
    const years = [2025, 2024, 2023, 2022, 2021, 2020]
    const filtered = vaultAlbums.filter(a => a.Year === vaultYear)

    return (
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.3rem', fontWeight: 600 }}>Vault</h2>
            <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: '#888' }}>
              {vaultYear === 2020 ? 'Top 50 · 2020' : `Top 100 · ${vaultYear}`}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {years.map(y => (
              <button
                key={y}
                onClick={() => setVaultYear(y)}
                style={{
                  background: vaultYear === y ? '#2d3748' : '#f5f5f5',
                  color: vaultYear === y ? '#e2e8f0' : '#555',
                  border: '1px solid #ddd',
                  borderRadius: '6px',
                  padding: '0.3rem 0.7rem',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: vaultYear === y ? 600 : 400,
                }}
              >
                {y}
              </button>
            ))}
          </div>
        </div>

        {vaultLoading && <p>Loading vault...</p>}

        {!vaultLoading && filtered.length > 0 && (
          <div className="billboard">
            <div className="billboard-header">
              <div className="rank-col">#</div>
              <div className="cover-col">Cover</div>
              <div className="artist-col">Artist</div>
              <div className="album-col">Album</div>
              <div className="review-col">Review</div>
            </div>
            {filtered.map((item) => {
              const spotifyUrl = item['Spotify URL']
              const href = spotifyUrl
                ? (spotifyUrl.startsWith('http') ? spotifyUrl : `https://${spotifyUrl}`)
                : null
              const albumKey = `${item.Artist}|${item.Album}`
              return (
                <div key={albumKey} className="billboard-row" style={{ cursor: 'default', height: 'auto', minHeight: '56px', padding: '10px 0', alignItems: 'flex-start' }}>
                  <div className="rank-col" style={{ fontWeight: 800, fontSize: '1.1rem', color: '#111', fontFamily: 'Georgia, serif', paddingTop: '2px' }}>
                    {Math.round(item.Rank)}
                  </div>
                  <div
                    className="cover-col"
                    style={{ cursor: 'pointer', position: 'relative' }}
                    onDoubleClick={() => {
                      const query = encodeURIComponent(`${item.Artist} ${item.Album} album cover`)
                      window.open(`https://www.google.com/search?q=${query}&tbm=isch`, '_blank')
                      onToggleCoverFix(albumKey)
                    }}
                  >
                    {getCoverUrl(item) ? (
                      <img src={getCoverUrl(item)} alt="cover" width="40" height="40" style={{ objectFit: 'cover', borderRadius: '4px' }} />
                    ) : (
                      <div style={{ width: 40, height: 40, background: '#f0f0f0', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem' }}>🎵</div>
                    )}
                  </div>
                  <div className="artist-col" style={{ fontWeight: 600, paddingTop: '2px' }}>{item.Artist}</div>
                  <div className="album-col" style={{ paddingTop: '2px' }}>
                    {href ? (
                      <a href={href} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', color: 'inherit' }}>
                        {item.Album}
                      </a>
                    ) : item.Album}
                  </div>
                  <div className="review-col" style={{ whiteSpace: 'normal', padding: '2px 8px 8px' }}>
                    {item['Mini-Review'] && (
                      <span style={{ display: 'block', color: '#444', fontStyle: 'italic', fontSize: '0.82rem', lineHeight: 1.4 }}>
                        {item['Mini-Review']}
                      </span>
                    )}
                    {item['Personal Anecdote'] && (
                      <span style={{ display: 'block', color: '#777', fontSize: '0.78rem', marginTop: '5px', lineHeight: 1.4, borderLeft: '2px solid #e0e0e0', paddingLeft: '6px' }}>
                        {item['Personal Anecdote']}
                      </span>
                    )}
                    {coverFixExpanded[albumKey] && (
                      <div style={{ marginTop: '8px', padding: '6px', background: '#f9f9f9', borderRadius: '6px' }}>
                        <input
                          type="text"
                          placeholder="Paste image URL..."
                          id={`cover-url-${albumKey}`}
                          style={{ width: '100%', fontSize: '0.75rem', padding: '4px', borderRadius: '4px', border: 'none', marginBottom: '4px' }}
                          onDoubleClick={e => e.stopPropagation()}
                        />
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            const url = document.getElementById(`cover-url-${albumKey}`).value
                            if (url) onSaveCover(item, url)
                          }}
                          style={{ width: '100%', background: '#1DB954', color: '#fff', border: 'none', borderRadius: '4px', padding: '4px', cursor: 'pointer', fontSize: '0.75rem' }}
                        >
                          Save
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    )
  }

  const DiscoverPage = () => {
    return (
      <div className="discover-page">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.3rem', fontWeight: 600 }}>Undrafted</h2>
            <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: '#888' }}>
              Albums the model scouted — rate to sign, ✕ to cut
            </p>
          </div>
          <button
            onClick={() => { setFeedAlbums([]); setFeedOffset(0); setFeedHasMore(true); loadMoreFeed(0, true) }}
            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '0.9rem', color: '#888', padding: '0.4rem' }}
          >
            Reshuffle
          </button>
        </div>

        {feedAlbums.length === 0 && feedLoading && <p>Loading your feed...</p>}
        {feedAlbums.length === 0 && !feedLoading && <p>No unrated albums found. You've rated everything! 🎉</p>}

        {feedAlbums.length > 0 && (
          <div className="cards-grid">
            {feedAlbums.map((album) => (
              <AlbumCard key={`${album.Artist}|${album.Album}`} album={album} {...cardProps} />
            ))}
          </div>
        )}

        {feedHasMore && !feedLoading && feedAlbums.length > 0 && (
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <button
              onClick={() => loadMoreFeed(feedOffset)}
              style={{ background: '#1DB954', color: '#fff', border: 'none', borderRadius: '8px', padding: '0.6rem 2rem', cursor: 'pointer', fontSize: '1rem', fontWeight: 500 }}
            >
              Load More
            </button>
          </div>
        )}

        {feedLoading && feedAlbums.length > 0 && (
          <p style={{ textAlign: 'center', marginTop: '1rem', color: '#888' }}>Loading more...</p>
        )}
      </div>
    )
  }

  return (
    <div className="container">
      <div className="tab-bar">
        <button className={`tab-button ${activeTab === 'weekly' ? 'active' : ''}`} onClick={() => setActiveTab('weekly')}>
          Weekly Predictions
        </button>
        <button className={`tab-button ${activeTab === 'top100' ? 'active' : ''}`} onClick={() => setActiveTab('top100')}>
          Top 100
        </button>
        <button className={`tab-button ${activeTab === 'discover' ? 'active' : ''}`} onClick={() => setActiveTab('discover')}>
          Undrafted
        </button>
        <button className={`tab-button ${activeTab === 'vault' ? 'active' : ''}`} onClick={() => setActiveTab('vault')}>
          Vault
        </button>
      </div>

      {activeTab === 'weekly' && (
        <>
          <div className="week-selector">
            <label htmlFor="week">Select Week: </label>
            <select id="week" value={selectedWeek} onChange={e => setSelectedWeek(e.target.value)}>
              {weeks.map(week => (
                <option key={week.formatted} value={week.formatted}>{week.formatted}</option>
              ))}
            </select>
          </div>
          {loading && <p>Loading...</p>}
          {!loading && sortedAlbums.length > 0 && (
            <div className="cards-grid">
              {sortedAlbums.map((album) => (
                <AlbumCard key={`${album.Artist}|${album.Album}`} album={album} {...cardProps} />
              ))}
            </div>
          )}
        </>
      )}

      {activeTab === 'top100' && <Top100Table />}
      {activeTab === 'discover' && <DiscoverPage />}
      {activeTab === 'vault' && <VaultPage
        coverFixExpanded={coverFixExpanded}
        onToggleCoverFix={handleToggleCoverFix}
        onSaveCover={handleSaveCover}
        getCoverUrl={getCoverUrl}
      />}
    </div>
  )
}

export default App
