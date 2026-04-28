import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DatasetsStats } from './datasets-stats';

describe('DatasetsStats', () => {
  let component: DatasetsStats;
  let fixture: ComponentFixture<DatasetsStats>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DatasetsStats],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasetsStats);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
