import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DatasetsPanel } from './datasets-panel';

describe('DatasetsPanel', () => {
  let component: DatasetsPanel;
  let fixture: ComponentFixture<DatasetsPanel>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DatasetsPanel],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasetsPanel);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
