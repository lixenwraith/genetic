package persistence

import "github.com/lixenwraith/genetic"

// SchemaVersion guards forward-incompatible layout changes
const SchemaVersion = 2

// PopulationDTO is the serializable population state. Tag is the caller's name
// for what the genes fit, a network's layout say, set before a save and
// compared after a load
type PopulationDTO struct {
	Version    int            `toml:"version" json:"version"`
	Tag        string         `toml:"tag" json:"tag"`
	Generation int            `toml:"generation" json:"generation"`
	Candidates []CandidateDTO `toml:"candidates" json:"candidates"`
}

// CandidateDTO is a serializable candidate
type CandidateDTO struct {
	Genes []float64 `toml:"genes" json:"genes"`
	Score float64   `toml:"score" json:"score"`
	Count int       `toml:"count" json:"count"`
}

// FromPool converts an engine snapshot to a DTO
func FromPool(pool *genetic.Pool[[]float64, float64]) PopulationDTO {
	if pool == nil {
		return PopulationDTO{Version: SchemaVersion}
	}

	dto := PopulationDTO{
		Version:    SchemaVersion,
		Generation: pool.Generation,
		Candidates: make([]CandidateDTO, len(pool.Members)),
	}
	for i, m := range pool.Members {
		dto.Candidates[i] = CandidateDTO{Genes: m.Data, Score: m.Score, Count: m.Count}
	}
	return dto
}

// ToPool converts a DTO to injectable candidates
func (dto PopulationDTO) ToPool() []genetic.Candidate[[]float64, float64] {
	out := make([]genetic.Candidate[[]float64, float64], len(dto.Candidates))
	for i, c := range dto.Candidates {
		out[i] = genetic.Candidate[[]float64, float64]{Data: c.Genes, Score: c.Score, Count: c.Count}
	}
	return out
}
